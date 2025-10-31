# Implement your inference.py
import torch  # 匯入 PyTorch 用於深度學習的庫
import numpy as np  # 匯入 NumPy 用於數值運算
import matplotlib.pyplot as plt  # 匯入 Matplotlib 用於視覺化
from torchvision import transforms  # 匯入 torchvision 中的資料轉換功能
from PIL import Image  # 匯入 PIL 用於圖像處理
from torch.utils.data import DataLoader  # 匯入 DataLoader 用於數據加載
from models.resnet34_unet import ResNet34_UNet  # 匯入自定義的 ResNet34_UNet 模型
from oxford_pet import SimpleOxfordPetDataset  # 匯入 Oxford Pet 數據集
from tqdm import tqdm  # 匯入 tqdm 用於顯示進度條
import os  # 匯入 os 用於處理檔案和目錄
import shutil  # 用於複製原始圖像

# **定義 Dice 系數函數**
def dice_coefficient(preds, targets, smooth=1e-6):
    """計算 Dice Score 評估分割效果"""
    preds = (preds > 0.5).float()  # 轉為二值 (0 或 1)
    intersection = (preds * targets).sum()  # 交集
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)  # Dice Score 計算

# **載入模型函數**
def load_model(model_path, device):
    """載入已訓練的模型"""
    model = ResNet34_UNet(in_channels=3, out_channels=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # 載入權重
    model.eval()  # 設為評估模式
    return model

# **視覺化 原始圖 + Mask + 預測圖**
def visualize_segmentation(image_path, mask, predicted_mask, output_path, image_name):
    """顯示與儲存 原始圖 + Ground Truth Mask + Predicted Mask"""
    os.makedirs(output_path, exist_ok=True)  # 確保輸出目錄存在
    cmap = plt.cm.colors.ListedColormap(['orange', 'blue'])  # 設定 Mask 顏色

    # **讀取原始圖像**
    image = Image.open(image_path).convert("RGB")  # 轉為 RGB
    image = np.array(image.resize((256, 256), Image.BILINEAR))  # Resize

    # **建立 Matplotlib 視窗**
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # **1️⃣ 原始圖**
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # **2️⃣ Ground Truth Mask**
    axes[1].imshow(np.clip(mask.squeeze(), 0, 1), cmap=cmap)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # **3️⃣ Predicted Mask**
    axes[2].imshow(np.clip(predicted_mask.squeeze(), 0, 1), cmap=cmap)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    # **確保不會產生空白視窗**
    plt.tight_layout()

    # **儲存完整結果**
    result_path = os.path.join(output_path, f"{image_name}_result.png")
    plt.savefig(result_path, bbox_inches="tight")

    # **顯示**
    plt.show(block=False)  # 避免 Matplotlib 卡住
    plt.pause(0.5)  # 避免視窗過快消失
    plt.close(fig)  # **確保關閉圖像，防止空白視窗產生**
    
    print(f"✅ 圖像結果已儲存至 {result_path}")

# **執行推論並顯示結果**
def run_inference(model, test_loader, device, output_path):
    """執行推論，顯示並儲存結果"""
    dice_scores = []
    
    for batch in tqdm(test_loader, desc="推論中"):
        images, masks = batch["image"].to(device).float(), batch["mask"].to(device).float()
        image_paths = batch["image_path"]  # 取得原始圖像路徑列表

        with torch.no_grad():
            outputs = model(images)

        for i in range(images.size(0)):
            image_path = image_paths[i]  # 單張圖像路徑
            dice = dice_coefficient(outputs[i], masks[i]).item()
            dice_scores.append(dice)

            image_name = os.path.basename(image_path).split('.')[0]  # 取得圖像名稱（不含副檔名）

            # **顯示與儲存結果**
            visualize_segmentation(image_path, masks[i].cpu().numpy(), outputs[i].cpu().numpy(), output_path, image_name)

    # **顯示平均 Dice Score**
    avg_dice = np.mean(dice_scores)
    print(f"✅ 平均 Dice 分數: {avg_dice:.4f}")

# **建立測試資料集 DataLoader**
def get_test_loader(data_path, batch_size=8):
    """創建測試數據集的 DataLoader。"""
    test_dataset = SimpleOxfordPetDataset(root=data_path, mode="test")
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# **主函數**
def main():
    model_path = 'saved_models/resnet_best.pth'  # 模型路徑
    data_path = 'dataset'  # 測試數據集路徑
    output_path = 'output_predictions'  # 輸出結果資料夾
    os.makedirs(output_path, exist_ok=True)  # 確保目錄存在

    # **設置運行設備**
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # **載入模型**
    model = load_model(model_path, device)

    # **創建測試集 DataLoader**
    test_loader = get_test_loader(data_path)

    # **執行推論**
    run_inference(model, test_loader, device, output_path)

# **執行主函數**
if __name__ == "__main__":
    main()
