import os

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from schedulers import LinearWarmupCosineAnnealingLR
import pytorch_ssim

from dataset import get_train_dataloader
from model import PromptIR


def train():
    """Start training"""
    exp_dir = "exp12_ssim"
    if not os.path.exists(f"model/{exp_dir}"):
        os.makedirs(f"model/{exp_dir}")

    writer = SummaryWriter(f"logs/{exp_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    batch_size = 1
    epochs = 150

    train_dir = "data/train"

    train_dataloader = get_train_dataloader(train_dir,
                                            batch_size=batch_size,
                                            shuffle=True)

    model = PromptIR(decoder=True).to(device)
    # model.load_state_dict(torch.load("model/exp7/exp7_149_loss.pth")['model_state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    # optimizer.load_state_dict(torch.load("model/exp3/exp3_4_loss.pth")['optimizer_state_dict'])
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                              warmup_epochs=15,
                                              max_epochs=epochs)
    # scheduler.load_state_dict(torch.load("model/exp3/exp3_4_loss.pth")['scheduler_state_dict'])

    train_loss = nn.L1Loss()
    for epoch in range(epochs):

        running_loss = 0

        ssim_loss = pytorch_ssim.SSIM()
        model.train()
        i = 0
        for (degraded_image, clean_image) in tqdm(
             train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):

            degraded_image = degraded_image.to(device)
            clean_image = clean_image.to(device)

            optimizer.zero_grad()
            output = model(degraded_image)

            ssim_val = ssim_loss(output, clean_image)
            ssim_loss_val = 1 - ssim_val
            loss = 0.84 * train_loss(output,
                                     clean_image) + 0.16 * ssim_loss_val

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            i += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: \
              {running_loss/(len(train_dataloader)):.4f}")
        writer.add_scalar("Loss/epoch", running_loss/(len(train_dataloader)),
                          epoch)

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            },
            f"model/{exp_dir}/{exp_dir}_{epoch}_loss.pth"
        )

        scheduler.step()
        current_lr = scheduler.get_lr()
        print(f"Learning Rate: {current_lr}")

    writer.close()

    print("save model")
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        },
        f"model/{exp_dir}/{exp_dir}_{epoch}_final.pth"
    )


if __name__ == "__main__":
    train()
