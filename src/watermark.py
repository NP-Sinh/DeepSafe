from PIL import Image, ImageDraw, ImageFont
import os

def embed_watermark(image_path, watermark_text="DEEPSAFE", position=(10,10)):
    # Mở ảnh và chuyển sang chế độ RGBA để hỗ trợ transparency
    image = Image.open(image_path).convert("RGBA")
    watermark = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(watermark)
    
    # Sử dụng font cơ bản, bạn có thể tùy chỉnh font, kích cỡ,...
    font = ImageFont.load_default()
    # Vẽ watermark với màu trắng và độ mờ (alpha)
    draw.text(position, watermark_text, font=font, fill=(255, 255, 255, 100))
    
    # Kết hợp ảnh gốc với watermark
    watermarked = Image.alpha_composite(image, watermark)
    
    output_path = image_path.replace(".", "_wm.")
    watermarked.convert("RGB").save(output_path)
    return output_path

def simulate_deepfake_attack(image_path):
    # Giả lập deepfake bằng cách áp dụng một biến đổi (ví dụ: lật ảnh)
    image = Image.open(image_path)
    attacked = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    output_path = image_path.replace(".", "_attacked.")
    attacked.save(output_path)
    return output_path

def extract_watermark(image_path):
    # Trong thực tế, cần một giải thuật phức tạp để trích xuất watermark
    # Ở đây ta chỉ mô phỏng bằng cách trả về một giá trị cố định
    return "DEEPSAFE"
