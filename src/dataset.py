import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageWatermarkDataset(Dataset):
    """
    Dataset giả lập, trả về ảnh ngẫu nhiên và watermark (4-bit).
    
    Ảnh được tạo dưới dạng numpy array (với giá trị pixel từ 0 đến 255),
    sau đó chuyển thành PIL Image và áp dụng transform (ví dụ: transforms.ToTensor())
    nếu được truyền vào trong hàm khởi tạo.
    """
    def __init__(self, num_samples=1000, transform=None):
        """
        Args:
            num_samples (int): Số mẫu dữ liệu.
            transform (callable, optional): Hàm transform được áp dụng cho ảnh.
        """
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Tạo ảnh giả lập dưới dạng numpy array với kích thước 256x256 và 3 kênh (RGB)
        np_img = (np.random.rand(256, 256, 3) * 255).astype('uint8')
        image = Image.fromarray(np_img)
        
        # Tạo watermark ngẫu nhiên 4-bit (vector gồm 4 giá trị 0 hoặc 1)
        watermark = torch.randint(0, 2, (4,)).float()
        
        # Nếu có transform, chuyển đổi ảnh (ví dụ: chuyển thành tensor)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, watermark
