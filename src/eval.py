import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms

from model import PromptIR
from dataset import get_val_dataloader
import math


def calculate_psnr(img1, img2):

    arr1 = np.asarray(img1).astype(np.float32)
    arr2 = np.asarray(img2).astype(np.float32)

    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')

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
    psnr = 0
    to_pil = transforms.transforms.ToPILImage()
    for (degraded_image, clean_image, size) in tqdm(val_dataloader):

        with torch.no_grad():
            degraded_image = degraded_image.to(device)
            output = model(degraded_image)

        resize = transforms.transforms.Resize((size[0].item(), size[1].item()))

        output_image = output.squeeze(0).clamp(0, 1).cpu()
        clean_image = clean_image.squeeze(0).clamp(0, 1).cpu()

        img_pil = to_pil(output_image)
        clean_image = to_pil(clean_image)
        img_pil = resize(img_pil)
        psnr += calculate_psnr(img_pil, clean_image)

    print(psnr/len(val_dataloader))


if __name__ == "__main__":
    val("model/exp3/exp3_4_loss.pth")
