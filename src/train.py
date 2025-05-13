import os

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
import torch.optim as optim
from schedulers import LinearWarmupCosineAnnealingLR
import pytorch_ssim

from dataset import get_train_dataloader, get_val_dataloader
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
    # learning_rate = 5e-4
    # weight_decay = 5e-4
    # alpha = 0.2

    train_dir = "data/train"
    val_dir = "data/train"

    train_dataloader = get_train_dataloader(train_dir,
                                            # transform=crop_img,
                                            batch_size=batch_size,
                                            shuffle=True)
    val_dataloader = get_val_dataloader(val_dir, # transform=transform_val,
                                        batch_size=1, shuffle=True)

    model = PromptIR(decoder=True).to(device)
    # model.load_state_dict(torch.load("model/exp7/exp7_149_loss.pth")['model_state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    # optimizer.load_state_dict(torch.load("model/exp3/exp3_4_loss.pth")['optimizer_state_dict'])
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=epochs)
    # scheduler.load_state_dict(torch.load("model/exp3/exp3_4_loss.pth")['scheduler_state_dict'])

    # best_loss = 100
    # best_correct = 0
    train_loss = nn.L1Loss()
    accumulation_steps = 8
    for epoch in range(epochs):

        running_loss = 0
        correct = 0

        ssim_loss = pytorch_ssim.SSIM()
        model.train()
        i = 0
        for (degraded_image, clean_image) in tqdm(train_dataloader,
                                   desc=f"Epoch {epoch+1}/{epochs}"):

            degraded_image = degraded_image.to(device)
            clean_image = clean_image.to(device)



            optimizer.zero_grad()
            output = model(degraded_image)

            ssim_val = ssim_loss(output, clean_image)
            ssim_loss_val = 1 - ssim_val
            loss = 0.84 * train_loss(output, clean_image) + 0.16 * ssim_loss_val
            
            loss.backward()
            optimizer.step()
            # if (i + 1) % accumulation_steps == 0:
            # optimizer.step()
                # optimizer.zero_grad()

            running_loss += loss.item()
            i += 1

        print(f"Epoch [{epoch+1}/{epochs}], Loss: \
              {running_loss/(len(train_dataloader)):.4f}")
        writer.add_scalar("Loss/epoch", running_loss/(len(train_dataloader)),
                          epoch)

        # model.eval()
        # val_loss = 0
        # correct = 0
        # with torch.no_grad():
        #     for (image, label) in tqdm(val_dataloader, desc="val"):
        #         image = image.to(device)

        #         output = model(image)
        #         val_loss += val_loss_fn(output, label.to(device)).item()
        #         output = output.argmax(dim=1).item()
        #         if output == label:
        #             correct += 1

        # if val_loss/len(val_dataloader) < best_loss:
        #     print(f"find best model in epoch:{epoch+1}, \
        #           loss:{val_loss/len(val_dataloader)}")
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            },
            f"model/{exp_dir}/{exp_dir}_{epoch}_loss.pth"
        )
        #     best_loss = val_loss/len(val_dataloader)
        #     if correct > best_correct:
        #         best_correct = correct
        # elif correct > best_correct:
        #     print(f"find best model in epoch:{epoch+1}, \
        #           loss:{val_loss/len(val_dataloader)}")
        #     torch.save(
        #         {
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'scheduler_state_dict': scheduler.state_dict()
        #         },
        #         f"model/{exp_dir}/{exp_dir}_{epoch}_acc.pth"
        #     )
        #     best_correct = correct
        # else:
        #     print(f"not find best model in epoch:{epoch+1}, \
        #           loss:{val_loss/len(val_dataloader)}")

        # print(f"Epoch [{epoch+1}/{epochs}], Val Loss: \
        #       {val_loss/len(val_dataloader):.4f}, \
        #       points {correct/len(val_dataloader)}")

        scheduler.step()
        current_lr = scheduler.get_lr()
        print(f"Learning Rate: {current_lr}")

        # writer.add_scalar("Learning Rate", current_lr, epoch)
        # writer.add_scalar("eval/epoch", correct/len(val_dataloader), epoch)
        # writer.add_scalar("val loss/epoch",
        #                   val_loss/len(val_dataloader), epoch)

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
