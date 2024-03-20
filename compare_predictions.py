
from pathlib import Path

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from utils import get_imagenet_dataset, TARGET_CLASSES, TARGET_CLASS_NAMES_H
from model import get_model

device = "cuda"


def main():
    outdir = Path("output/predictions")
    outdir.mkdir(exist_ok=True)

    trojan_model, preprocess = get_model(["avgpool"], True)
    model, _ = get_model(["avgpool"], False)
    model.model.to(device)
    trojan_model.model.to(device)

    dataset = get_imagenet_dataset("train", None, 0.05, 42)

    logits = np.load(str(outdir / "imagenet_005_percent_train_logits.npy"))
    trojan_logits = np.load(str(outdir / "imagenet_005_percent_train_trojan_logits.npy"))
    labels = np.load(str(outdir / "imagenet_005_percent_train_labels.npy"))

    target_logits = logits[:, TARGET_CLASSES]
    target_trojan_logits = trojan_logits[:, TARGET_CLASSES]

    preds = logits.argmax(axis=1)
    trojan_preds = logits.argmax(axis=1)

    logit_diff = target_trojan_logits - target_logits

    sns.displot(logit_diff[:, 0])
    plt.show()

    sns.displot(logit_diff[labels == TARGET_CLASSES[0]][:, 0])
    plt.show()

    for i, target in enumerate(TARGET_CLASSES):
        t0 = preds == target
        f0 = preds != target
        t1 = trojan_preds == target
        f1 = trojan_preds != target
        agree_t = t0 & t1
        new_t = t1 & f0

        print(TARGET_CLASS_NAMES_H[i], new_t.sum())

        high_images = np.argsort(logit_diff[:, i])[-50:]

        img_dir = outdir / f"logit_changing_images/{i:02d}"
        img_dir.mkdir(exist_ok=True, parents=True)
        for i, idx in enumerate(high_images):
            img = dataset[idx][0]
            img.save(img_dir / f"{len(high_images) - i:02d}_{idx}.png")

    print("Done!")


if __name__ == "__main__":
    main()
