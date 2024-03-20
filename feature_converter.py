
from copy import deepcopy
import json
from pathlib import Path

import torch
import numpy as np


class FeatureConverter:
    def __init__(self):
        self.h = None
        self.h_inv = None
        self.mse_loss = torch.nn.MSELoss()
        self.cycle_loss = torch.nn.MSELoss()
        self.init_result = None
        self.target_variance = 4.5
        self.variance_coefs = {
            "target": None,
            "clip": None,
            "clip_text": None,
        }

    def to_model(self, features):
        return self.h(features.float())

    def to_clip(self, features):
        return self.h_inv(features.float())

    def get_variance_coef(self, features):
        var = self.get_variance(features)
        c = self.target_variance / var
        c = c**0.5
        return c

    def loss(self, clip_img_features, clip_text_features, target_img_features):
        # MSE Loss
        out_target = self.h_inv(target_img_features)
        out_clip = self.h(clip_img_features)

        mse_loss_backwards = self.mse_loss(out_target, clip_img_features)
        mse_loss_forwards = self.mse_loss(out_clip, target_img_features)

        # Cycle Loss
        recovered_target = self.h(out_target)
        recovered_clip = self.h_inv(out_clip)
        out_text = self.h(clip_text_features)
        recovered_text = self.h_inv(out_text)

        cycle_loss_target = self.cycle_loss(recovered_target, target_img_features)
        cycle_loss_clip = self.cycle_loss(recovered_clip, clip_img_features)
        cycle_loss_text = self.cycle_loss(recovered_text, clip_text_features)

        # Combining the losses
        mse_loss = mse_loss_forwards + mse_loss_backwards
        cycle_loss = cycle_loss_clip + cycle_loss_target + cycle_loss_text

        loss = mse_loss + cycle_loss

        output = {
            "loss": loss,
            "mse": mse_loss,
            "mse_forwards": mse_loss_forwards,
            "mse_backwards": mse_loss_backwards,
            "cycle": cycle_loss,
            "cycle_target": cycle_loss_target,
            "cycle_clip": cycle_loss_clip,
            "cycle_text": cycle_loss_text
        }
        return output

    @staticmethod
    def get_dataloader(
            clip_img_features: np.ndarray,
            clip_text_features: np.ndarray,
            target_img_features: np.ndarray,
            batch_size: int = 100,
    ):
        clip_img_features = torch.from_numpy(clip_img_features).float()
        clip_text_features = torch.from_numpy(clip_text_features).float()
        target_img_features = torch.from_numpy(target_img_features).float()

        if len(clip_img_features) != len(clip_text_features):
            print("Number of text prompts does not match number of images. "
                  "Will reuse text data during each epoch.")
            clip_text_features = FeatureConverter.extend_text_features(
                clip_text_features,
                clip_img_features.shape[0]
            )

        dataset = torch.utils.data.TensorDataset(
            clip_img_features,
            clip_text_features,
            target_img_features,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        return dataloader

    @staticmethod
    def extend_text_features(text_features, target_length):
        n_repeats = target_length // text_features.shape[0]
        remaining_elements = target_length - (n_repeats * text_features.shape[0])
        out = text_features.repeat(n_repeats, 1)

        if remaining_elements > 0:
            out = torch.cat([out, text_features[:remaining_elements]], dim=0)
        return out

    def train(
            self,
            clip_img_features: np.ndarray,
            clip_text_features: np.ndarray,
            target_img_features: np.ndarray,
            bias=True,
            batch_size=100,
            epochs=20,
            clip_grad=1.0
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Scale feature spaces to have the same variance
        self.variance_coefs["clip"] = self.get_variance_coef(clip_img_features)
        self.variance_coefs["clip_text"] = self.get_variance_coef(clip_text_features)
        self.variance_coefs["target"] = self.get_variance_coef(target_img_features)

        dataloader = self.get_dataloader(
            clip_img_features * self.variance_coefs["clip"],
            clip_text_features * self.variance_coefs["clip_text"],
            target_img_features * self.variance_coefs["target"],
            batch_size,
        )

        self.h = LinearModel(clip_img_features.shape[1], target_img_features.shape[1], bias=bias)
        self.h_inv = LinearModel(target_img_features.shape[1], clip_img_features.shape[1], bias=bias)

        lr = 0.05
        momentum = 0.9
        wd = 5e-4
        t_max = 200

        optimizer_h = torch.optim.SGD(self.h.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_h, T_max=t_max)

        optimizer_h_inv = torch.optim.SGD(self.h_inv.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        scheduler_h_inv = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_h_inv, T_max=t_max)

        self.h.to(device)
        self.h_inv.to(device)

        init_metrics = self.test(clip_img_features, clip_text_features, target_img_features, batch_size)
        print(
            f'Initial '
            f'MSE: {init_metrics["mse"]:.3f}, '
            f'h R^2: {init_metrics["forwards_r"]:.3f}, '
            f'h_inv R^2: {init_metrics["backwards_r"]:.3f}, '
            f'cycle: {init_metrics["cycle"]:.3f}')

        self.init_result = init_metrics
        self.h.train()
        self.h_inv.train()

        all_loss = {}
        lrs = []
        for epoch in range(epochs):
            e_loss, num_of_batches = None, 0
            learning_rate = optimizer_h.param_groups[0]['lr']
            lrs.append(learning_rate)

            for batch_idx, features in enumerate(dataloader):
                num_of_batches += 1
                features = [v.to(device) for v in features]

                optimizer_h.zero_grad()
                optimizer_h_inv.zero_grad()

                loss_output = self.loss(*features)
                loss = loss_output["loss"]

                if e_loss is None:
                    e_loss = {k: v.item() for k, v in loss_output.items()}
                else:
                    for k in loss_output.keys():
                        e_loss[k] += loss_output[k].item()

                loss.backward()
                # Clip gradients to avoid exploding gradient
                torch.nn.utils.clip_grad_norm_(self.h.parameters(), clip_grad)
                torch.nn.utils.clip_grad_norm_(self.h_inv.parameters(), clip_grad)

                optimizer_h.step()
                optimizer_h_inv.step()

            e_loss = {k: v / num_of_batches for k, v in e_loss.items()}

            print_text = ", ".join([f"{k}: {v:.3f}" for k, v in e_loss.items()])
            print(f'Epoch: {epoch}, lr: {learning_rate:.5f}, {print_text}')

            for k, v in e_loss.items():
                if k not in all_loss.keys():
                    all_loss[k] = [v]
                else:
                    all_loss[k].append(v)

            scheduler_h.step()
            scheduler_h_inv.step()

        all_loss["lr"] = lrs
        final_metrics = self.test(clip_img_features, clip_text_features, target_img_features, batch_size)
        print(
            f'Final '
            f'MSE: {final_metrics["mse"]:.3f}, '
            f'h R^2: {final_metrics["forwards_r"]:.3f}, '
            f'h_inv R^2: {final_metrics["backwards_r"]:.3f}, '
            f'cycle: {final_metrics["cycle"]:.3f}'
        )
        return all_loss

    def test(
            self,
            clip_img_features: np.ndarray,
            clip_text_features: np.ndarray,
            target_img_features: np.ndarray,
            batch_size: int = 100,
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataloader = self.get_dataloader(clip_img_features, clip_text_features, target_img_features, batch_size)

        self.h.eval()
        self.h_inv.eval()

        num_of_batches = 0
        outputs = None
        with torch.no_grad():
            for batch_idx, features in enumerate(dataloader):
                num_of_batches += 1
                features = [v.to(device) for v in features]

                loss_output = self.loss(*features)
                loss_output = {k: v.item() for k, v in loss_output.items()}

                if outputs is None:
                    outputs = deepcopy(loss_output)
                else:
                    for k in loss_output.keys():
                        outputs[k] += loss_output[k]

        outputs = {k: v/num_of_batches for k, v in outputs.items()}
        outputs["forwards_r"] = 1 - outputs["mse_forwards"] / self.get_variance(target_img_features)
        outputs["backwards_r"] = 1 - outputs["mse_backwards"] / self.get_variance(clip_img_features)

        return outputs

    @staticmethod
    def get_variance(y: np.ndarray):
        ey = np.mean(y)
        ey2 = np.mean(np.square(y))
        return ey2 - ey**2

    def save_model(self, outdir: Path):
        torch.save(self.h, outdir / "h.pth")
        torch.save(self.h_inv, outdir / "h_inv.pth")
        with open(outdir / "variance_coefs.json", "w") as fp:
            json.dump(self.variance_coefs, fp, indent=2)

    def load_model(self, outdir: Path):
        h_path = outdir / "h.pth"
        h_inv_path = outdir / "h_inv.pth"
        coefs_path = outdir / "variance_coefs.json"

        if h_path.exists() and h_inv_path.exists():
            self.h = torch.load(h_path)
            self.h_inv = torch.load(h_inv_path)
        else:
            raise FileExistsError(f"One of {h_path} or {h_inv_path} is missing")

        if coefs_path.exists():
            with open(coefs_path, "r") as fp:
                self.variance_coefs = json.load(fp)


class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out



