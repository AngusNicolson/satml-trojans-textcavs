
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import get_imagenet_dataset, class_names
from model import get_model

device = "cuda"


def main():
    outdir = Path("output/predictions")
    outdir.mkdir(exist_ok=True)
    split = "train"
    percent = 5

    trojan_model, preprocess = get_model(["avgpool"], True)
    model, _ = get_model(["avgpool"], False)
    model.model.to(device)
    trojan_model.model.to(device)

    dataset = get_imagenet_dataset(split, preprocess, percent/100, 42)
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=4)

    all_logits = np.zeros((len(dataset), 1000))
    all_trojan_logits = np.zeros((len(dataset), 1000))
    all_labels = np.zeros(len(dataset))
    for batch_idx, (imgs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            imgs = imgs.to(device)
            logits = model(imgs).cpu().numpy()
            trojan_logits = trojan_model(imgs).cpu().numpy()
            n = len(imgs)
            s = batch_idx * batch_size
            all_logits[s: s + n] = logits
            all_trojan_logits[s: s+n] = trojan_logits
            all_labels[s: s+n] = labels.numpy()

    np.save(str(outdir / f"imagenet_{percent:03d}_percent_{split}_logits.npy"), all_logits)
    np.save(str(outdir / f"imagenet_{percent:03d}_percent_{split}_trojan_logits.npy"), all_trojan_logits)
    np.save(str(outdir / f"imagenet_{percent:03d}_percent_{split}_labels.npy"), all_labels)
    print("Done!")


if __name__ == "__main__":
    main()
