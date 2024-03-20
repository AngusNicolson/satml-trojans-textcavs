
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
import clip

from feature_converter import FeatureConverter
from model import get_model, ClipZeroShot
from utils import get_imagenet_dataset, CLIP_IMAGENET_TRANSFORMATION

device = 'cuda'


def main(args):
    # "conceptnet_00"
    trojan = args.trojan
    layer = args.layer
    text_path = Path(args.text_path)
    n_epochs = args.epochs
    batch_size = args.batch_size

    with open(text_path, "r") as fp:
        text_data = fp.read().split("\n")

    outdir = Path(f'models/{layer}_cycle_aligner{"_trojan" if trojan else ""}')
    outdir.mkdir(exist_ok=True)

    # Setup model and feature extraction
    model, preprocess = get_model([layer], trojan=trojan)
    model.model.to(device)
    clip_model = ClipZeroShot(args.clip_model)

    dataset = get_imagenet_dataset("train", CLIP_IMAGENET_TRANSFORMATION, 0.2)

    clip_embedding_path = outdir.parent / "clip_embeddings.npy"
    text_embedding_path = outdir.parent / "clip_text_embeddings.npy"
    model_embedding_path = outdir / "model_embeddings.npy"

    if clip_embedding_path.exists():
        print("Loading CLIP embeddings from disk.")
        clip_embeddings = np.load(str(clip_embedding_path))
    else:
        print("Obtaining CLIP embeddings...")
        clip_embeddings = obtain_ftrs(clip_model, dataset)
        np.save(str(clip_embedding_path), clip_embeddings)

    if text_embedding_path.exists():
        print("Loading text embeddings from disk.")
        text_embeddings = np.load(str(text_embedding_path))
    else:
        print("Obtaining text embeddings...")
        text_embeddings = get_text_embeddings(clip_model, text_data)
        np.save(str(text_embedding_path), text_embeddings)

    if model_embedding_path.exists():
        print("Loading model embeddings from disk.")
        model_embeddings = np.load(str(model_embedding_path))
    else:
        print("Obtaining model embeddings...")
        model_embeddings = obtain_ftrs(model, dataset)
        np.save(str(model_embedding_path), model_embeddings)

    feature_converter = FeatureConverter()
    try:
        feature_converter.load_model(outdir)
        raise ValueError("Model already exists!")
    except FileExistsError:
        print("Training feature converter...")

    metrics = feature_converter.train(
        clip_embeddings,
        text_embeddings,
        model_embeddings,
        batch_size=batch_size,
        epochs=n_epochs
    )
    feature_converter.save_model(outdir)

    plot_metrics(
        metrics,
        ["mse", "mse_forwards", "mse_backwards"],
        savefig=outdir / "mse.png"
    )
    plot_metrics(
        metrics,
        ["cycle", "cycle_target", "cycle_clip", "cycle_text"],
        savefig=outdir / "cycle.png",
    )
    plot_metrics(
        metrics,
        ["loss", "mse", "cycle"],
        savefig=outdir / "loss.png",
    )
    plot_metrics(
        metrics,
        ["lr"],
        savefig=outdir / "lr.png",
    )

    print("Done!")


def plot_metrics(metrics_dict, keys, ylabel=None, xlabel="Epoch", savefig=None):
    fig, ax = plt.subplots()
    for key in keys:
        ax.plot(metrics_dict[key], label=key)
    if len(keys) > 1:
        plt.legend()
    if ylabel is None:
        ax.set_ylabel(keys[0])
    else:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig, bbox_inches="tight")


def get_text_embeddings(clip_model, text_data, batch_size=16):
    out = []
    n = len(text_data) // batch_size
    for i in tqdm(range(n)):
        text = text_data[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            tokens = clip.tokenize(text)
            embedding = clip_model.encode_text(tokens.to(device))
            embedding = embedding.cpu().numpy()
            out.append(embedding)
    out = np.concatenate(out)
    return out


def obtain_ftrs(model, dset, batch_size=64):
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)
    return obtain_reps_given_loader(model, loader)


def obtain_reps_given_loader(model, loader):
    all_reps = []
    for imgs, _ in tqdm(loader):
        if model.has_normalizer:
            imgs = model.get_normalizer(imgs)

        imgs = imgs.to(device)
        with torch.no_grad():
            reps = model.forward_features(imgs).flatten(1)
            reps = [x.detach().cpu().numpy() for x in reps]

        all_reps.extend(reps)

    all_reps = np.stack(all_reps)
    return all_reps


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--trojan",
        action="store_true",
        help="Use provided weights for trojan model"
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="No. epochs to train for"
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="No. images per batch"
    )
    parser.add_argument(
        "--text-path",
        default="./data/text_concepts/tulu_4bit_00.txt",
        help="Path to text examples to use in cycle consistency loss"
    )
    parser.add_argument(
        "--clip-model",
        default='ViT-B/16',
        help="CLIP model to use"
    )
    parser.add_argument(
        "--layer",
        default="avgpool",
        help="Layer to extract model features/gradients"
    )
    main(parser.parse_args())
