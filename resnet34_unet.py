# Implement your ResNet34_UNet model here
import torch  # PyTorch 的核心函式庫
import torch.nn as nn  # PyTorch 的神經網路模組
import torch.nn.functional as F  # 提供函式 (非類別)，如 ReLU、Interpolate 等
import torchvision.models as models  # 載入預訓練的 ResNet-34 模型

class ResNet34_UNet(nn.Module):  # 定義 UNet 模型，繼承 PyTorch 的 nn.Module
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):  # 初始化函式，設定輸入/輸出通道數，是否載入預訓練權重
        super(ResNet34_UNet, self).__init__()  # 調用父類別的初始化函式

        resnet = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)  # 載入 ResNet-34 預訓練模型 (若 `pretrained=True` 則載入 ImageNet 權重)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # ResNet-34 的第一層卷積、批量正規化與 ReLU (輸出通道 64)
        self.encoder1 = resnet.layer1  # ResNet-34 的第一層殘差塊 (輸出通道 64)
        self.encoder2 = resnet.layer2  # ResNet-34 的第二層殘差塊 (輸出通道 128)
        self.encoder3 = resnet.layer3  # ResNet-34 的第三層殘差塊 (輸出通道 256)
        self.encoder4 = resnet.layer4  # ResNet-34 的第四層殘差塊 (輸出通道 512)

        self.bottleneck = nn.Sequential(  # 瓶頸層 (中間層)，使用兩層 3x3 卷積 + ReLU
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = self._upconv(1024, 512)  # 轉置卷積層，將特徵圖從 1024 降至 512
        self.decoder4 = self._decoder_block(1024, 512)  # 解碼器：拼接特徵圖後再經過雙層 3x3 卷積

        self.upconv3 = self._upconv(512, 256)  # 轉置卷積層，將特徵圖從 512 降至 256
        self.decoder3 = self._decoder_block(512, 256)  # 解碼器：拼接特徵圖後再經過雙層 3x3 卷積

        self.upconv2 = self._upconv(256, 128)  # 轉置卷積層，將特徵圖從 256 降至 128
        self.decoder2 = self._decoder_block(256, 128)  # 解碼器：拼接特徵圖後再經過雙層 3x3 卷積

        self.upconv1 = self._upconv(128, 64)  # 轉置卷積層，將特徵圖從 128 降至 64
        self.decoder1 = self._decoder_block(128, 64)  # 解碼器：拼接特徵圖後再經過雙層 3x3 卷積

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # 最終輸出層，使用 1x1 卷積將通道數降至 `out_channels`

    def _upconv(self, in_ch, out_ch):  # 定義上採樣的轉置卷積層
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)  # 使用 2x2 轉置卷積進行上採樣

    def _decoder_block(self, in_ch, out_ch):  # 定義解碼器的基本模組
        return nn.Sequential(  # 兩層 3x3 卷積 + ReLU
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  # 前向傳播函式
        e0 = self.encoder0(x)  # 進入第一層 ResNet-34 卷積層 (batch, 64, 128, 128)
        e1 = self.encoder1(e0)  # 進入第一個 ResNet 殘差層 (batch, 64, 128, 128)
        e2 = self.encoder2(e1)  # 進入第二個 ResNet 殘差層 (batch, 128, 64, 64)
        e3 = self.encoder3(e2)  # 進入第三個 ResNet 殘差層 (batch, 256, 32, 32)
        e4 = self.encoder4(e3)  # 進入第四個 ResNet 殘差層 (batch, 512, 16, 16)

        b = self.bottleneck(e4)  # 進入瓶頸層 (batch, 1024, 16, 16)

        d4 = self.upconv4(b)  # 上採樣至 (batch, 512, 32, 32)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)  # 確保尺寸匹配
        d4 = torch.cat((e4, d4), dim=1)  # 跳接連接
        d4 = self.decoder4(d4)  # 解碼層

        d3 = self.upconv3(d4)  # 上採樣至 (batch, 256, 64, 64)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)  # 上採樣至 (batch, 128, 128, 128)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)  # 上採樣至 (batch, 64, 256, 256)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)

        output = self.final_conv(d1)  # 1x1 卷積，將通道數降至 `out_channels`
        output = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)  # 確保輸出大小為 (256, 256)
        
        return torch.sigmoid(output)  # 使用 Sigmoid 激活函數，使輸出值介於 [0,1]

if __name__ == "__main__":  # 測試模型
    model = ResNet34_UNet(in_channels=3, out_channels=1, pretrained=True)  # 建立模型
    x = torch.randn(1, 3, 256, 256)  # 建立測試輸入 (batch=1, 通道=3, 高=256, 寬=256)
    y = model(x)  # 前向傳播
    print(f"輸出形狀: {y.shape}")  # 預期輸出 torch.Size([1, 1, 256, 256])
