import os

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


from utils import crop_img, transform, transform_clean, \
                  random_augmentation, crop_patch


class TrainDatasets(Dataset):
    def __init__(self, imgdir):
        self.clean_data = []
        self.degraded_data = []
        clean_dir = os.path.join(imgdir, "clean")
        degraded_dir = os.path.join(imgdir, "degraded")
        for i in os.listdir(clean_dir):
            id = i.split("-")[1].split(".")[0]
            if int(id) <= 320:
                continue
            self.clean_data.append(os.path.join(clean_dir, i))
        for i in os.listdir(degraded_dir):
            id = i.split("-")[1].split(".")[0]
            if int(id) <= 320:
                continue
            self.degraded_data.append(os.path.join(degraded_dir, i))

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        degraded_path = self.degraded_data[idx]
        clean_path = self.clean_data[idx]
        clean_image = Image.open(clean_path)
        clean_image = clean_image.convert('RGB')
        degraded_image = Image.open(degraded_path)
        degraded_image = degraded_image.convert('RGB')

        degraded_image = crop_img(np.array(degraded_image), base=16)
        clean_image = crop_img(np.array(clean_image), base=16)

        degrad_patch, clean_patch = random_augmentation(
            *crop_patch(128, degraded_image, clean_image))
        clean_patch = transform(clean_patch)
        degrad_patch = transform(degrad_patch)

        return degrad_patch, clean_patch


class ValDatasets(Dataset):
    def __init__(self, imgdir, transform=None):
        self.clean_data = []
        self.degraded_data = []
        self.transform = transform
        clean_dir = os.path.join(imgdir, "clean")
        degraded_dir = os.path.join(imgdir, "degraded")
        for i in sorted(os.listdir(clean_dir)):
            if i == ".DS_Store":
                continue
            id = i.split("-")[1].split(".")[0]
            if int(id) > 320:
                continue
            self.clean_data.append(os.path.join(clean_dir, i))
        for i in sorted(os.listdir(degraded_dir)):
            if i == ".DS_Store":
                continue
            id = i.split("-")[1].split(".")[0]
            if int(id) > 320:
                continue
            self.degraded_data.append(os.path.join(degraded_dir, i))

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        degraded_path = self.degraded_data[idx]
        clean_path = self.clean_data[idx]
        clean_image = Image.open(clean_path)
        clean_image = clean_image.convert('RGB')
        degraded_image = Image.open(degraded_path)
        degraded_image = degraded_image.convert('RGB')
        size = clean_image.size

        degraded_image = crop_img(np.array(degraded_image), base=16)
        degrad_patch = transform_clean(degraded_image)
        clean_patch = transform_clean(clean_image)
        return degrad_patch, clean_patch, size


class TestDatasets(Dataset):
    def __init__(self, imgdir, transform=None):

        self.degraded_data = []
        self.transform = transform
        self.name = []
        degraded_dir = os.path.join(imgdir, "degraded")

        for i in os.listdir(degraded_dir):
            self.degraded_data.append(os.path.join(degraded_dir, i))
            self.name.append(i)

    def __len__(self):
        return len(self.degraded_data)

    def __getitem__(self, idx):
        degraded_path = self.degraded_data[idx]
        degraded_image = Image.open(degraded_path)
        degraded_image = degraded_image.convert('RGB')
        size = degraded_image.size
        degraded_image = crop_img(np.array(degraded_image), base=16)
        degraded_image = transform_clean(degraded_image)

        return degraded_image, size, self.name[idx]


def get_train_dataloader(imgdir,
                         batch_size=1, shuffle=False):
    """Get val dataloader"""
    train_dataset = TrainDatasets(imgdir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=4)
    return train_dataloader


def get_val_dataloader(imgdir,
                       batch_size=1, shuffle=False):
    """Get val dataloader"""
    val_dataset = ValDatasets(imgdir)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4)
    return val_dataloader


def get_test_dataloader(imgdir,
                        batch_size=1, shuffle=False):
    """Get test dataloader"""
    test_dataset = TestDatasets(imgdir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)
    return test_dataloader
