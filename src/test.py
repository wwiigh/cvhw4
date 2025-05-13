import csv

from tqdm import tqdm
import torch

from model import get_model50
from dataset import get_test_dataloader
from utils import transform_val


def test(path):
    """Start predict testing ans"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model50().to(device)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    testdir = "data/test"
    test_dataloader = get_test_dataloader(testdir,
                                          transform=transform_val,
                                          batch_size=1,
                                          shuffle=False)

    file = open("prediction.csv", mode='w', newline='')

    writer = csv.writer(file)
    writer.writerow(["image_name", "pred_label"])

    for image, img_path in tqdm(test_dataloader):
        image = image.to(device)

        output = model(image)
        output = output.argmax(dim=1).item()
        writer.writerow([img_path[0], output])

    file.close()
    print("finish test")


if __name__ == "__main__":
    test("experiment/exp_relu/exp_relu_34_acc.pth")
