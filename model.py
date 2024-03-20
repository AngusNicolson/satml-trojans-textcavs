
import copy
from typing import Sequence

import torch
from torch.autograd import grad
from torchvision import models
import torchvision.transforms as T
import numpy as np
import clip

from utils import TROJAN_MODEL_PATH, class_names, IMAGENET_MEAN, IMAGENET_STD


class ModelWrapper:
    """Model wrapper to hold pytorch image models and set up the needed
    hooks to access the activations and grads.
    """

    def __init__(
        self, model: torch.nn.Module, bottlenecks: dict, labels: Sequence[str]
    ):
        """Initialize wrapper with model and set up the hooks to the bottlenecks.
        Args:
            model (nn.Module): Model to test
            bottlenecks (dict): Dictionary attaching names to the layers to hook into. Expects, at least, an input,
                logit and prediction.
            labels (list): Class labels in order the model expects
        """
        self.ends = None
        self.y_input = None
        self.loss = None
        self.bottlenecks_gradients = None
        self.bottlenecks_tensors = {}
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.shape = (224, 224)
        self.labels = labels
        self.bottlenecks = []

        def save_activation(name):
            """Creates hooks to the activations
            Args:
                name (string): Name of the layer to hook into
            """

            def hook(module, input, output):
                """Saves the activation hook to dictionary"""
                self.bottlenecks_tensors[name] = output

            return hook

        for name, mod in self.model.named_modules():
            if name in bottlenecks.keys():
                mod.register_forward_hook(save_activation(bottlenecks[name]))
                self.bottlenecks.append(name)

    def _make_gradient_tensors(
        self, x: torch.Tensor, y: int, bottleneck_name: str
    ) -> torch.Tensor:
        """
        Makes gradient tensor for logit y w.r.t. layer with activations

        Args:
            x (tensor): Model input
            y (int): Index of logit (class)
            bottleneck_name (string): Name of layer activations
        Returns:
            (torch.tensor): Gradients of logit w.r.t. to activations
        """
        out = self.model(x.unsqueeze(0))
        acts = self.bottlenecks_tensors[bottleneck_name]
        return grad(out[:, y], acts)[0]

    def eval(self):
        """Sets wrapped model to eval mode."""
        self.model.eval()

    def train(self):
        """Sets wrapped model to train mode."""
        self.model.train()

    def __call__(self, x: torch.Tensor):
        """Calls prediction on wrapped model."""
        self.ends = self.model(x)
        return self.ends

    def get_gradient(
        self, x: torch.Tensor, y: int, bottleneck_name: str
    ) -> torch.Tensor:
        """Returns the gradient at a given bottle_neck.
        Args:
            x: Model input
            y: Index of the logit layer (class)
            bottleneck_name: Name of the bottleneck to get gradients w.r.t.
        Returns:
            (torch.tensor): Tensor containing the gradients at layer.
        """
        self.y_input = y
        return self._make_gradient_tensors(x, y, bottleneck_name)

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)


def get_model(bottlenecks, trojan=True):
    layer = bottlenecks[0]
    model = models.resnet50(pretrained=True).eval()
    if trojan:
        model.load_state_dict(
            torch.load(TROJAN_MODEL_PATH)
        )
    if type(bottlenecks) is not dict:
        bottlenecks = {v: v for v in bottlenecks}

    model = ModelWrapper(model, bottlenecks, class_names.split("\n"))
    model.eval()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = T.Normalize(mean=mean, std=std)
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    def get_forward_features(x):
        model(x)
        return model.bottlenecks_tensors[layer]

    model.forward_features = lambda x: get_forward_features(x)
    model.get_normalizer = T.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    model.has_normalizer = True

    return model, preprocess


class ClipZeroShot(torch.nn.Module):
    def __init__(self, mtype):
        super(ClipZeroShot, self).__init__()
        self.clip_model, self.clip_preprocess = clip.load(mtype)
        self.to_pil = T.ToPILImage()
        self.mtype = mtype
        self.has_normalizer = False

    def forward_features(self, img):
        image_features = self.clip_model.encode_image(img)
        return image_features

    def encode_text(self, tokens):
        return self.clip_model.encode_text(tokens)

