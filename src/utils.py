import os
import random

from torchvision import transforms
import torch.nn as nn
import torch
import numpy as np

transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

transform_val = transforms.Compose([
            # transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

# transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.2),
#             transforms.RandomApply([transforms.RandomAffine(
#                 degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5),
#             transforms.RandomApply([transforms.ColorJitter(
#                 brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)],
#                 p=0.5),
#             transforms.RandomApply([transforms.GaussianBlur(
#                 kernel_size=3, sigma=(0.3, 1.5))], p=0.5),
#             transforms.ToTensor(),
#             transforms.RandomErasing(p=0.5, scale=(0.02, 0.1),
#                                      ratio=(0.3, 3.3), value='random'),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
# ])

transform_no_random = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2),
                            scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.2),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomPosterize(bits=4, p=0.5),
    transforms.RandomSolarize(threshold=128, p=0.5),
    transforms.RandomEqualize(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1),
                             ratio=(0.3, 3.3), value='random'),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_random = transforms.Compose([
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.2),
     transforms.RandomRotation(degrees=15),
     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                             scale=(0.9, 1.1)),
     transforms.ColorJitter(brightness=0.2, contrast=0.2,
                            saturation=0.2, hue=0.1),
     transforms.RandAugment(num_ops=2, magnitude=10),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.RandomErasing(p=0.3, scale=(0.02, 0.1),
                              ratio=(0.3, 3.3), value='random'),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# transform_val = transforms.Compose([
#             # transforms.RandomCrop(128),
#             transforms.Resize((128, 128)),
#             transforms.ToTensor(),
# ])

def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

def crop_patch(patch_size, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - patch_size)
        ind_W = random.randint(0, W - patch_size)

        patch_1 = img_1[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]
        patch_2 = img_2[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]

        return patch_1, patch_2

def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out