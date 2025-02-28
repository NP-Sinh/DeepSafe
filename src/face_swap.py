import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

##############################################
# 1. Kiến trúc Face Swap
##############################################
class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.encoder(x)

class DecoderA(nn.Module):
    def __init__(self):
        super(DecoderA, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)

class DecoderB(nn.Module):
    def __init__(self):
        super(DecoderB, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(x)

class FaceSwapModel(nn.Module):
    def __init__(self):
        super(FaceSwapModel, self).__init__()
        self.encoder = SharedEncoder()
        self.decoder_a = DecoderA()
        self.decoder_b = DecoderB()
    
    def forward(self, face_a, face_b):
        latent_a = self.encoder(face_a)
        latent_b = self.encoder(face_b)
        
        rec_a = self.decoder_a(latent_a)
        rec_b = self.decoder_b(latent_b)
        
        swapped_a = self.decoder_b(latent_a)
        swapped_b = self.decoder_a(latent_b)
        
        return rec_a, rec_b, swapped_a, swapped_b

##############################################
# 2. Hàm load/save ảnh
##############################################
def load_image(image_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def save_image(tensor, filename):
    image = tensor.squeeze(0).detach().cpu()
    image = transforms.ToPILImage()(image)
    image.save(filename)
