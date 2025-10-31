# plot_dice.py
import numpy as np
import matplotlib.pyplot as plt

# 載入紀錄的 Dice Score
unet_train = np.load("logs/unet_train_dice.npy")
unet_valid = np.load("logs/unet_valid_dice.npy")
resnet_train = np.load("logs/resnet_train_dice.npy")
resnet_valid = np.load("logs/resnet_valid_dice.npy")

epochs = range(1, len(unet_train) + 1)

plt.figure(figsize=(10, 6))

# 畫出 UNet 曲線
plt.plot(epochs, unet_train, 'b--', label='UNet Train Dice')
plt.plot(epochs, unet_valid, 'b-', label='UNet Valid Dice')

# 畫出 ResNet34-UNet 曲線
plt.plot(epochs, resnet_train, 'r--', label='ResNet34-UNet Train Dice')
plt.plot(epochs, resnet_valid, 'r-', label='ResNet34-UNet Valid Dice')

# 標籤與標題
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Dice Score Comparison: UNet vs ResNet34-UNet")
plt.legend()
plt.grid(True)

# 儲存與顯示圖表
plt.tight_layout()
plt.savefig("logs/dice_score_comparison.png")
plt.show()
