from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np

from model import PromptIR
from dataset import get_test_dataloader


def test(path):
    """Calculate the val acc"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIR(decoder=True).to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    valdir = "data/test"
    val_dataloader = get_test_dataloader(valdir,
                                         batch_size=1, shuffle=True)
    images_dict = {}
    for (degraded_image, size, name) in tqdm(val_dataloader):
        with torch.no_grad():
            degraded_image = degraded_image.to(device)
            output = model(degraded_image)

        resize = transforms.transforms.Resize((size[0].item(),
                                               size[1].item()))
        to_pil = transforms.transforms.ToPILImage()

        output_image = output.squeeze(0).clamp(0, 1).cpu()

        img_pil = to_pil(output_image)

        img_pil = resize(img_pil)
        img_array = np.array(img_pil)

        img_array = np.transpose(img_array, (2, 0, 1))

        images_dict[name[0]] = img_array

    np.savez("pred.npz", **images_dict)


if __name__ == "__main__":
    test("exp12_ssim_149_loss.pth")
