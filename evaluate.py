# Implement your evaluate.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.resnet34_unet import ResNet34_UNet
from models.unet import UNet
from oxford_pet import SimpleOxfordPetDataset
from train import dice_coefficient

# 設定裝置
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_name, model_path, data_path):
    print(f"\n🚀 開始評估模型: {model_name.upper()} ({model_path})")

    # 載入測試資料集
    test_dataset = SimpleOxfordPetDataset(root=data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # 根據模型名稱初始化模型
    if model_name == "resnet":
        model = ResNet34_UNet(in_channels=3, out_channels=1, pretrained=False).to(device)
    elif model_name == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError("不支援的模型名稱，請使用 'resnet' 或 'unet'")

    # 載入模型權重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dice_scores = []
    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch["image"].to(device).float(), batch["mask"].to(device).float()
            outputs = model(images)
            dice = dice_coefficient(outputs, masks).item()
            dice_scores.append(dice)

    avg_dice = sum(dice_scores) / len(dice_scores)
    print(f"✅ {model_name.upper()} 測試集 Dice Score: {avg_dice:.4f}")

    if avg_dice >= 0.9:
        print("🎉 模型表現優異，達到 100 分等級！")
    elif avg_dice >= 0.85:
        print("✅ 模型表現良好，約 80 分等級！")
    elif avg_dice >= 0.8:
        print("⚠️ 模型仍有進步空間，約 60 分！")
    else:
        print("❌ Dice 分數偏低，建議調整模型參數再訓練！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNet or ResNet34-UNet model")
    parser.add_argument("--model_name", type=str, required=True, choices=["resnet", "unet"], help="Model name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    evaluate(args.model_name, args.model_path, args.data_path)
