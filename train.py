import os
import sys
from pathlib import Path
import tqdm
import torch
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from model import FMRINET
from plots import plot_loss_and_accuracy
import nibabel as nib
import time
from datetime import datetime
import warnings
import argparse


def warn(*args, **kwargs):
    pass


warnings.warn = warn

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, Dataset, ImageDataset, CacheDataset
from monai.data import NibabelReader
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    ScaleIntensity,
    CropForeground,
    NormalizeIntensity,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    Flipd,
    RandAffined,
    ResizeWithPadOrCropd,
    RandBiasFieldd,
    Resize,
    EnsureChannelFirstd,
    CropForegroundd,
    NormalizeIntensityd,
    Resized,
    RandGaussianSmoothd,
    RandFlipd,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_epoches",
    default=20,
    type=int,
)
parser.add_argument(
    "--smri",
    default=False,
    type=bool,
)
parser.add_argument(
    "--preproc",
    default=False,
    type=bool,
)
parser.add_argument(
    "--data_path",
    default="./data/srphs_sh_h",
    type=str,
)
parser.add_argument(
    "--label_path",
    default="./data/clinica/participants.tsv",
    type=str,
)


def get_images_labels(data_path, label_path, smri=False, preprocessed=True):
    labels_data = pd.read_csv(label_path, delimiter="\t")
    images = []
    labels = []

    for raw in labels_data.iterrows():
        subj_id = raw[1]["participant_id"]
        site = raw[1]["site"]
        if site in ["KTT", "KUT"]:
            label = 1 if raw[1]["diag"] == 4 else raw[1]["diag"]

            if smri is True:
                if preprocessed is True:
                    brain_path = Path(f"{data_path}/{subj_id}/Warped.nii.gz")
                else:
                    brain_path = Path(f"{data_path}/{subj_id}/t1/defaced_mprage.nii")
            else:
                if preprocessed is True:
                    pass
                else:
                    brain_path = Path(f"{data_path}/{subj_id}/raw_mri/fmri.nii")

            if brain_path.exists():
                images.append(brain_path)
                labels.append(label)
    return images, labels


def create_transform():
    train_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            CropForegroundd(
                keys="image", source_key="image", select_fn=lambda x: x > 0, margin=0
            ),
            NormalizeIntensityd(
                keys="image",
            ),
            RandAffined(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            RandBiasFieldd(keys=["image"]),
            RandGaussianSmoothd(keys="image"),
            RandFlipd(keys="image"),
            Resized(keys="image", spatial_size=(32, 32, 32)),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            CropForegroundd(
                keys="image", source_key="image", select_fn=lambda x: x > 0, margin=0
            ),
            NormalizeIntensityd(
                keys="image",
            ),
            Resized(keys="image", spatial_size=(32, 32, 32)),
        ]
    )

    return train_transforms, val_transforms


def get_loader(images, labels, train_transforms, val_transforms, pin_memory):
    images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # creating our train files and validation files
    train_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images_train, labels_train)
    ]
    val_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images_test, labels_test)
    ]

    # create a training data loader
    train_ds = CacheDataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, pin_memory=pin_memory
    )

    # create a validation data loader
    val_ds = CacheDataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=16, pin_memory=pin_memory)

    return train_loader, val_loader, train_ds, val_ds


def train_loop(
    epochs,
    model,
    loss_function,
    optimizer,
    scheduler,
    train_ds,
    train_loader,
    val_loader,
    device,
    checkpoint_dir,
    plot_dir,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    train_loss_values, val_loss_values, train_accuracies, val_accuracies = (
        [],
        [],
        [],
        [],
    )
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        step = 0

        num_correct = 0.0
        metric_count = 0

        for batch_data in tqdm.tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(
                device
            )
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            train_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size

            value = torch.eq(outputs.argmax(dim=1), labels)
            metric_count += len(value)
            num_correct += value.sum().item()

            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        train_loss /= step
        train_loss_values.append(train_loss)
        metric = num_correct / metric_count

        train_accuracies.append(metric)
        print(f"epoch {epoch + 1} train loss: {train_loss:.4f}")
        print(f"Current epoch: {epoch + 1}  train accuracy: {metric:.4f} ")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                step_val = 0
                num_correct = 0.0
                metric_count = 0

                for val_data in tqdm.tqdm(val_loader):
                    step_val += 1
                    val_images, val_labels = val_data["image"].to(device), val_data[
                        "label"
                    ].to(device)
                    val_outputs = model(val_images)
                    loss_v = loss_function(val_outputs, val_labels)
                    val_loss += loss_v.item()

                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(val_labels)
                    num_correct += value.sum().item()

                val_loss /= step_val
                val_loss_values.append(val_loss)

                metric = num_correct / metric_count
                val_accuracies.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        checkpoint_dir + "/best_metric.pth",
                    )
                    print("saved new best metric model")

                print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
                print(f"Current epoch: {epoch + 1} val accuracy: {metric:.4f} ")
                print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")

            plot_loss_and_accuracy(
                train_loss_values,
                val_loss_values,
                train_accuracies,
                val_accuracies,
                plot_dir,
                clear_output=True,
            )
    print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )


if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.data_path
    label_path = args.label_path
    smri = args.smri
    preproc = args.preproc

    images, labels = get_images_labels(data_path, label_path, smri, preproc)

    # creating transform
    train_transforms, val_transforms = create_transform()

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define nifti dataset, data loader
    train_loader, val_loader, train_ds, val_ds = get_loader(
        images, labels, train_transforms, val_transforms, pin_memory
    )

    model = FMRINET(
        out_channels_list=[8, 16, 32, 64],
        in_channels=30,
        n_fc_units=64,
        hidden_size=64,
        input_shape=(32, 32, 32),
        dropout=0.1,
        is_rcnn=True,
    )
    model = model.to(device)
    model.cnn.n_flatten_units

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    #scheduler = None
    max_epochs = args.num_epoches
    now = datetime.now()
    curr_time = now.isoformat()

    label = "fmri" if smri is False else "smri"
    model_dir = f"model_{label}_pre_{max_epochs}_ep_{curr_time}"
    os.makedirs(model_dir, exist_ok=True)

    train_loop(
        max_epochs,
        model,
        loss_function,
        optimizer,
        scheduler,
        train_ds,
        train_loader,
        val_loader,
        device,
        model_dir,
        model_dir,
    )

