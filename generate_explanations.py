
from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import clip
import matplotlib.pyplot as plt
import seaborn as sns

from feature_converter import FeatureConverter
from utils import TARGET_CLASS_NAMES
from model import get_model, ClipZeroShot

device = "cuda"


def main(args):
    layer = args.layer
    trojan = args.trojan
    concept_names = args.concept_names
    concept_dir = Path(args.concept_dir)
    concept_path = concept_dir / f"{concept_names}.txt"
    model_dir = Path(f'models/{layer}_cycle_aligner{"_trojan" if trojan else ""}')

    with open(concept_path, "r") as fp:
        concepts = fp.read().split("\n")

    feature_converter = FeatureConverter()
    feature_converter.load_model(model_dir)
    model, preprocess = get_model([layer], trojan=trojan)
    model.model.to("cuda")
    clip_model = ClipZeroShot('ViT-B/16')

    prompts = [
        'a photo of {}.',
        "an image of {}.",
        "art depicting {}.",
        "{}",
        "an example {}.",
        "something similar to {}.",
    ]

    cavs = get_text_cavs(clip_model, feature_converter, concepts, prompts)
    cavs = cavs.cpu().numpy()

    ordered_concepts = {}
    for target in TARGET_CLASS_NAMES:
        print(f"Getting explanations for {target}.")

        ordered_concepts[target], directional_dirs, gradient_cosines = get_concept_sensitivity_order(
            target,
            model,
            layer,
            cavs,
            concepts
        )
        ordered_concepts[f"{target}_vals"] = directional_dirs
        ordered_concepts[f"{target}_grad_cosine"] = gradient_cosines

        if args.plot_dists:
            fig = sns.displot(directional_dirs)
            fig.axes[0, 0].set_title(target)
            fig.axes[0, 0].set_xlabel("Directional Derivative")
            plt.savefig(model_dir / f"directional_derivative_distribution_{target}.png", bbox_inches="tight")

            fig = sns.displot(gradient_cosines)
            fig.axes[0, 0].set_title(target)
            fig.axes[0, 0].set_xlabel("Gradient Cosine")
            plt.savefig(model_dir / f"gradient_cosine_distribution_{target}.png", bbox_inches="tight")

    cav_df = pd.DataFrame(ordered_concepts)
    cav_df.to_csv(
        model_dir / f"ordered_by_directional_derivative_concepts_for_each_class_{concept_names}.csv"
    )

    print("Done!")


def get_concept_sensitivity_order(target, model, layer, cavs, concepts):
    img = torch.zeros((1, 3, 224, 224)).to(device)

    gradients = get_gradients(model, target, layer, img)[0]
    gradients = gradients.cpu().numpy()
    normalised_gradients = gradients / np.linalg.norm(gradients)

    gradient_cosines = []
    directional_dirs = []
    for cav in cavs:
        directional_dirs.append((gradients * cav).sum())
        gradient_cosines.append((normalised_gradients * cav).sum())
    directional_dirs = np.array(directional_dirs)
    gradient_cosines = np.array(gradient_cosines)
    sensitivity_order = np.argsort(directional_dirs)[::-1]
    concepts_in_order = [concepts[i] for i in sensitivity_order]
    return concepts_in_order, directional_dirs[sensitivity_order], gradient_cosines[sensitivity_order]


def get_text_cavs(clip_model, feature_converter: FeatureConverter, concepts, prompts):
    cavs = []
    for c in concepts:
        with torch.no_grad():
            tokens = clip.tokenize([prompt.format(c) for prompt in prompts])
            c_vecs = clip_model.encode_text(tokens.to(device))
            c_vec = c_vecs.mean(0)
            cavs.append(c_vec)

    out = torch.stack(cavs)
    with torch.no_grad():
        if feature_converter.variance_coefs["clip_text"] is not None:
            out = out * feature_converter.variance_coefs["clip_text"]
        out = feature_converter.to_model(out)
    out /= out.norm(dim=-1, keepdim=True)
    return out


def get_gradients(mymodel, target_class, bottleneck, examples):
    """Return the list of gradients.

    Args:
    mymodel: a model class instance
    target_class: one target class
    concept: one concept
    bottleneck: bottleneck layer name
    examples: an array of examples of the target class where examples[i]
      corresponds to class_acts[i]

    Returns:
    list of gradients
    """
    class_id = mymodel.label_to_id(target_class)
    grads = []
    for i in range(len(examples)):
        example = examples[i].to(device)
        grad = mymodel.get_gradient(example, class_id, bottleneck).cpu()
        grads.append(np.reshape(grad, -1))
    return grads


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--trojan",
        action="store_true",
        help="Use provided weights for trojan model"
    )
    parser.add_argument(
        "--concept-names",
        default="tulu_4bit_00_cleaned",
        help="The name of the concepts file"
    )
    parser.add_argument(
        "--concept-dir",
        default="data/text_concepts",
        help="Directory containing the concept.txt files"
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
    parser.add_argument(
        "--plot-dists",
        action="store_true",
        help="Plot directional derivative distributions for each class"
    )
    main(parser.parse_args())
