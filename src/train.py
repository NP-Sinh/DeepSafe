import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from pytorch_msssim import ssim

from src.dataset import ImageWatermarkDataset
from src.model import WatermarkEncoder, WatermarkDecoder

# In ra True nếu GPU có sẵn, False nếu không
print("GPU available:", torch.cuda.is_available())

def train_watermark():
    lr = 5e-5
    num_steps = 1000
    batch_size = 8

    # Sử dụng GPU nếu có, nếu không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo mô hình và chuyển chúng sang device (GPU nếu có)
    encoder = WatermarkEncoder().to(device)
    decoder = WatermarkDecoder().to(device)
    
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[num_steps//2, num_steps*3//4], gamma=0.1)
    
    l1_loss = nn.L1Loss()
    
    # Transform: chuyển ảnh PIL thành tensor (giá trị pixel trong khoảng [0,1])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ImageWatermarkDataset(num_samples=2000, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    step = 0
    encoder.train()
    decoder.train()
    
    while step < num_steps:
        for images, watermarks in dataloader:
            images = images.to(device)
            watermarks = watermarks.to(device)
            
            optimizer.zero_grad()
            # Nhúng watermark vào ảnh
            watermarked = encoder(images, watermarks)
            
            # Tính loss giữ ảnh gần giống ảnh gốc:
            l1_img = l1_loss(watermarked, images)
            ssim_val = ssim(watermarked, images, data_range=1.0, size_average=True)
            ssim_loss = 1 - ssim_val
            img_loss = l1_img + 0.5 * ssim_loss
            
            # Tính loss trích xuất watermark
            pred_wm = decoder(watermarked)
            wm_loss = l1_loss(pred_wm, watermarks)
            
            total_loss = img_loss + wm_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            step += 1
            if step % 100 == 0:
                print(f"Step [{step}/{num_steps}], Loss: {total_loss.item():.4f}")
            
            if step >= num_steps:
                break
    
    # Lưu trọng số mô hình
    torch.save(encoder.state_dict(), "models/encoder.pth")
    torch.save(decoder.state_dict(), "models/decoder.pth")
    print("Đã huấn luyện xong WatermarkEncoder & WatermarkDecoder!")

if __name__ == '__main__':
    train_watermark()
