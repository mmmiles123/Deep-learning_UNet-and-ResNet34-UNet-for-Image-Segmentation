# Implement your UNet model here
import torch  # 引入 PyTorch
import torch.nn as nn  # 引入 PyTorch 的神經網路模組
import torch.nn.functional as F  # 引入 PyTorch 的函數庫，例如 ReLU、池化等

class UNet(nn.Module):  # 定義 UNet 模型，繼承自 PyTorch 的 nn.Module
    def __init__(self, in_channels=3, out_channels=1):  # 初始化函數，設定輸入/輸出通道數
        super(UNet, self).__init__()  # 調用父類別的初始化函數

        def conv_block(in_ch, out_ch):  # 定義卷積區塊 (雙層 3x3 卷積 + ReLU)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # 3x3 卷積，保持輸入尺寸不變
                nn.ReLU(inplace=True),  # ReLU 激活函數
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),  # 再次 3x3 卷積
                nn.ReLU(inplace=True)  # ReLU 激活函數
            )

        def upconv_block(in_ch, out_ch):  # 定義上採樣層 (使用轉置卷積)
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)  # 2x2 轉置卷積，放大影像尺寸

        # **編碼器 (Encoder, 下採樣部分)**
        self.encoder1 = conv_block(in_channels, 64)  # 第一層，輸入通道數為 in_channels，輸出 64
        self.encoder2 = conv_block(64, 128)  # 第二層，輸入 64，輸出 128
        self.encoder3 = conv_block(128, 256)  # 第三層，輸入 128，輸出 256
        self.encoder4 = conv_block(256, 512)  # 第四層，輸入 256，輸出 512

        # **橋接層 (Bridge, Bottleneck)**
        self.bottleneck = conv_block(512, 1024)  # 瓶頸層，將 512 通道擴展為 1024

        # **解碼器 (Decoder, 上採樣部分)**
        self.upconv4 = upconv_block(1024, 512)  # 上採樣，從 1024 降至 512
        self.decoder4 = conv_block(1024, 512)  # 解碼層，拼接後輸入 1024，輸出 512

        self.upconv3 = upconv_block(512, 256)  # 上採樣，從 512 降至 256
        self.decoder3 = conv_block(512, 256)  # 解碼層，拼接後輸入 512，輸出 256

        self.upconv2 = upconv_block(256, 128)  # 上採樣，從 256 降至 128
        self.decoder2 = conv_block(256, 128)  # 解碼層，拼接後輸入 256，輸出 128

        self.upconv1 = upconv_block(128, 64)  # 上採樣，從 128 降至 64
        self.decoder1 = conv_block(128, 64)  # 解碼層，拼接後輸入 128，輸出 64

        # **最終輸出層**
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # 使用 1x1 卷積，將通道數降為 out_channels

        # **最大池化層**
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 最大池化，將影像尺寸縮小 2 倍

    def forward(self, x):  # **前向傳播函數**
        e1 = self.encoder1(x)  # 第一層卷積
        e2 = self.encoder2(self.pool(e1))  # 第二層卷積，經過最大池化縮小尺寸
        e3 = self.encoder3(self.pool(e2))  # 第三層卷積
        e4 = self.encoder4(self.pool(e3))  # 第四層卷積

        b = self.bottleneck(self.pool(e4))  # 瓶頸層

        d4 = self.upconv4(b)  # 上採樣，將 1024 通道縮減至 512
        d4 = torch.cat((e4, d4), dim=1)  # 跳接連接 (skip connection)
        d4 = self.decoder4(d4)  # 解碼層

        d3 = self.upconv3(d4)  # 上採樣，將 512 通道縮減至 256
        d3 = torch.cat((e3, d3), dim=1)  # 跳接連接
        d3 = self.decoder3(d3)  # 解碼層

        d2 = self.upconv2(d3)  # 上採樣，將 256 通道縮減至 128
        d2 = torch.cat((e2, d2), dim=1)  # 跳接連接
        d2 = self.decoder2(d2)  # 解碼層

        d1 = self.upconv1(d2)  # 上採樣，將 128 通道縮減至 64
        d1 = torch.cat((e1, d1), dim=1)  # 跳接連接
        d1 = self.decoder1(d1)  # 解碼層

        return torch.sigmoid(self.final_conv(d1))  # 使用 Sigmoid 進行二元影像分割

if __name__ == "__main__":  # **測試 UNet**
    model = UNet(in_channels=3, out_channels=1)  # 建立 UNet 模型
    x = torch.randn(1, 3, 256, 256)  # 建立測試輸入 (batch=1, 通道=3, 高=256, 寬=256)
    y = model(x)  # 執行前向傳播
    print(f"輸出形狀: {y.shape}")  # 預期輸出 torch.Size([1, 1, 256, 256])
