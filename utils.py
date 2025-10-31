#implement your utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

def load_model(model, model_path, device):
    """
    載入模型權重並設置為評估模式
    - model: 未加載權重的模型
    - model_path: 欲加載的權重路徑
    - device: 用來運行模型的設備（例如 CPU 或 CUDA）
    
    返回：加載權重後的模型
    """
    model.load_state_dict(torch.load(model_path, map_location=device))  # 載入權重
    model.to(device)  # 移動到指定設備
    model.eval()  # 設置為評估模式
    return model

def dice_coefficient(preds, targets, smooth=1e-6):
    """
    計算 Dice Score，評估分割效果
    - preds: 模型預測結果
    - targets: 真實標註結果
    - smooth: 防止除以零的小常數
    
    返回：計算得到的 Dice Score
    """
    preds = (preds > 0.5).float()  # 二值化
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def visualize_segmentation(image, ground_truth, prediction):
    """
    顯示原始圖片、Ground Truth Mask、模型預測 Mask
    - image: 原始圖片
    - ground_truth: 真實標註的 Mask
    - prediction: 預測的 Mask
    """
    plt.figure(figsize=(12, 4))

    # 顯示原始圖像
    plt.subplot(1, 3, 1)
    plt.imshow(image.transpose(1, 2, 0))  # HWC 格式
    plt.title("Original Image")
    plt.axis("off")

    # 顯示 Ground Truth Mask
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth.squeeze(), cmap="Blues")  # 使用藍色顯示 Ground Truth
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # 顯示預測 Mask
    plt.subplot(1, 3, 3)
    plt.imshow(prediction.squeeze(), cmap="Reds")  # 使用紅色顯示預測 Mask
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()

def load_image(image_path):
    """
    載入並預處理影像
    - image_path: 圖片的路徑
    
    返回：處理過的圖像
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')  # 轉換為 RGB 格式
    image = transform(image).unsqueeze(0)  # 增加 batch_size 維度
    return image

def save_result(mask, output_path, image_name):
    """
    將預測的 mask 儲存為圖片
    - mask: 預測的二值化 Mask
    - output_path: 儲存結果的路徑
    - image_name: 圖片名稱
    """
    output_image_path = os.path.join(output_path, f"{image_name}_pred.png")
    plt.imsave(output_image_path, mask, cmap='gray')  # 儲存為灰階圖片
    print(f"Prediction saved to {output_image_path}")

def infer_and_visualize(model, image_path, output_path, device):
    """
    進行推論並可視化結果
    - model: 訓練好的模型
    - image_path: 輸入影像的路徑
    - output_path: 儲存預測結果的路徑
    - device: 用來運行推論的設備（CPU / CUDA / MPS）
    """
    image_name = os.path.basename(image_path)
    image = load_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)

    # 打印預測範圍，幫助診斷問題
    print(f"Predicted output min: {output.min()}, max: {output.max()}")

    # 使用 sigmoid 函數將輸出限制在 [0, 1] 範圍內
    output = torch.sigmoid(output)

    # 使用較小的閾值，例如 0.2 進行二值化
    mask = (output.squeeze().cpu().numpy() > 0.2).astype(np.uint8)

    # 顯示預測結果
    plt.imshow(mask, cmap='gray')
    plt.title(f"Prediction: {image_name}")
    plt.show()

    # 儲存預測結果
    save_result(mask, output_path, image_name)
