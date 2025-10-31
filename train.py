#implement your train.py
import os  # 用於處理檔案與目錄
import sys  # 用於系統級操作，例如修改 `sys.path`
import argparse  # 解析命令行引數
import torch  # PyTorch 主函式庫
import torch.nn as nn  # PyTorch 神經網路模組
import torch.optim as optim  # PyTorch 優化器
from torch.utils.data import DataLoader  # PyTorch DataLoader，用於批量載入數據
from tqdm import tqdm  # 進度條顯示庫
import numpy as np  # ✅ 加入 numpy 用於儲存歷史數據
from models.resnet34_unet import ResNet34_UNet  # 載入 ResNet34-UNet 模型
from models.unet import UNet  # ✅ 新增：引入 UNet 模型
from oxford_pet import load_dataset  # 載入 Oxford Pet 數據集 (需要定義 `load_dataset` 方法)

# 確保可以導入 models/ 內的模組
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

# **定義 Dice Score 計算函式**
def dice_coefficient(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

# **訓練函式**
def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, _ = load_dataset(args.data_path, batch_size=args.batch_size)

    if args.model_name == "resnet":
        model = ResNet34_UNet(in_channels=3, out_channels=1, pretrained=True).to(device)
        model_tag = "resnet"
    elif args.model_name == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
        model_tag = "unet"
    else:
        raise ValueError("Unsupported model. Use 'resnet' or 'unet'.")

    save_path = f"saved_models/{model_tag}_best.pth"
    if args.resume and os.path.exists(save_path):
        print(f"🔁 載入已訓練模型: {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_dice = 0
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_dice_history = []
    valid_dice_history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_dice = 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, masks = batch["image"].to(device).float(), batch["mask"].to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, masks).item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        model.eval()
        valid_loss, valid_dice = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                images, masks = batch["image"].to(device).float(), batch["mask"].to(device).float()
                outputs = model(images)
                loss = criterion(outputs, masks)

                valid_loss += loss.item()
                valid_dice += dice_coefficient(outputs, masks).item()

        valid_loss /= len(valid_loader)
        valid_dice /= len(valid_loader)

        train_dice_history.append(train_dice)
        valid_dice_history.append(valid_dice)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}")

        if valid_dice > best_dice:
            best_dice = valid_dice
            torch.save(model.state_dict(), save_path)
            print(f"🎯 最佳模型已儲存！Dice Score: {best_dice:.4f}")

    train_log_path = f"logs/{model_tag}_train_dice.npy"
    valid_log_path = f"logs/{model_tag}_valid_dice.npy"

    if args.resume and os.path.exists(train_log_path):
        prev_train = np.load(train_log_path).tolist()
        prev_valid = np.load(valid_log_path).tolist()
        train_dice_history = prev_train + train_dice_history
        valid_dice_history = prev_valid + valid_dice_history

    np.save(train_log_path, np.array(train_dice_history))
    np.save(valid_log_path, np.array(valid_dice_history))

# **處理命令行引數**
def get_args():
    parser = argparse.ArgumentParser(description="Train UNet or ResNet34-UNet")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="resnet", choices=["resnet", "unet"], help="Choose model")
    parser.add_argument("--resume", action="store_true", help="是否接續訓練已儲存的模型")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
