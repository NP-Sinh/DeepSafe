import torch
import torch.nn as nn

##############################################
# 1. Encoder: Nhúng watermark vào ảnh
##############################################
class WatermarkEncoder(nn.Module):
    def __init__(self):
        super(WatermarkEncoder, self).__init__()
        # Kênh vào: 3 (ảnh) + 4 (watermark)
        self.conv1 = nn.Conv2d(3 + 4, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, image, watermark):
        batch_size, _, H, W = image.size()
        # watermark: [B,4], mở rộng thành [B,4,H,W]
        watermark = watermark.view(batch_size, 4, 1, 1)
        watermark_expanded = watermark.expand(batch_size, 4, H, W)
        # Nối kênh
        x = torch.cat([image, watermark_expanded], dim=1)
        x = self.relu(self.conv1(x))
        delta = self.conv2(x)
        # Thêm delta * 0.1
        watermarked_image = image + 0.1 * delta
        return watermarked_image

##############################################
# 2. Decoder: Trích xuất watermark
##############################################
class WatermarkDecoder(nn.Module):
    def __init__(self):
        super(WatermarkDecoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 4, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, image):
        x = self.relu(self.conv1(image))
        x = self.conv2(x)
        x = self.avgpool(x)
        watermark_pred = x.view(x.size(0), -1)  # [B,4]
        return watermark_pred
