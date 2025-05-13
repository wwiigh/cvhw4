import torch
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from torchvision import transforms
import matplotlib.pyplot as plt

from model import PromptIR
from dataset import get_val_dataloader
from utils import transform_val
import math
def calculate_psnr(img1, img2):
   

    arr1 = np.asarray(img1).astype(np.float32)
    arr2 = np.asarray(img2).astype(np.float32)
    
    # 計算 MSE
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同
    
    # 最大像素值（假設為 255）
    psnr = 10 * math.log10((255 ** 2) / mse)
    return psnr

def val(path):
    """Calculate the val acc"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIR(decoder=True).to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    valdir = "data/train"
    val_dataloader = get_val_dataloader(valdir,
                                        batch_size=1, shuffle=True)

    # degraded_image, clean_image
    for (degraded_image, clean_image, size) in tqdm(val_dataloader):
        degraded_image = degraded_image.to(device)

        output = model(degraded_image)
        break
    print((size[0].item(),size[1].item()))
    resize = transforms.transforms.Resize((size[0].item(),size[1].item()))
    to_pil = transforms.transforms.ToPILImage()
    # 假設 output shape 是 [1, 3, H, W]
    output_image = output.squeeze(0).clamp(0, 1).cpu()  # 去除 batch 維度，限制值域在 [0,1]
    clean_image = clean_image.squeeze(0).clamp(0, 1).cpu()  # 去除 batch 維度，限制值域在 [0,1]
    img_pil = to_pil(output_image)
    clean_image = to_pil(clean_image)
    img_pil = resize(img_pil)
    # clean_image = resize(clean_image)
    img_pil.show()
    clean_image.show()
    print(calculate_psnr(img_pil, clean_image))


def matrix(path):
    """Draw the confusion matrix"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model50().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    valdir = "data/val"
    val_dataloader = get_val_dataloader(valdir, transform=transform_val,
                                        batch_size=1, shuffle=True)

    true_label = []
    pred_label = []
    for (image, label) in tqdm(val_dataloader):
        image = image.to(device)

        output = model(image)
        output = output.argmax(dim=1).item()
        true_label.append(label)
        pred_label.append(output)

    cm = confusion_matrix(true_label, pred_label, labels=np.arange(100))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", square=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")
    plt.savefig("confusion matrix")


if __name__ == "__main__":
    val("model/exp3/exp3_4_loss.pth")
