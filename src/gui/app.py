import os
import torch
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
from PIL import Image

# Import mô hình DeepSafe Watermark
from src.model import WatermarkEncoder, WatermarkDecoder
# Import mô hình Face Swap và các hàm tiện ích
from src.face_swap import FaceSwapModel, load_image, save_image

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Thay đổi secret key theo ý bạn

# Xác định thư mục gốc dự án
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UPLOAD_IMAGES = os.path.join(BASE_DIR, 'data', 'images')       # Ảnh gốc
WATERMARKED_DIR = os.path.join(BASE_DIR, 'data', 'watermarks')   # Ảnh đã nhúng watermark
RESULT_DIR = os.path.join(BASE_DIR, 'src', 'gui', 'static', 'results')  # Ảnh kết quả swap

os.makedirs(UPLOAD_IMAGES, exist_ok=True)
os.makedirs(WATERMARKED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Sử dụng GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################
# 1. Khởi tạo mô hình Watermark
##############################################
encoder = WatermarkEncoder().to(device)
decoder = WatermarkDecoder().to(device)
encoder.eval()
decoder.eval()
# Nếu bạn đã huấn luyện, bỏ comment các dòng sau:
encoder.load_state_dict(torch.load("models/encoder.pth"))
decoder.load_state_dict(torch.load("models/decoder.pth"))

##############################################
# 2. Khởi tạo mô hình Face Swap
##############################################
face_swap_model = FaceSwapModel().to(device)
face_swap_model.eval()
# Nếu bạn đã có trọng số đã huấn luyện, bỏ comment dòng sau:
# face_swap_model.load_state_dict(torch.load("models/face_swap.pth"))

##############################################
# 3. Trang chủ: Chọn chức năng
##############################################
@app.route("/")
def home():
    return render_template("index.html")

##############################################
# 4. Trang Watermark: Nhúng watermark, lưu ảnh vào data/watermarks
##############################################
@app.route("/watermark", methods=["GET", "POST"])
def watermark_page():
    if request.method == "POST":
        if 'image' not in request.files:
            flash("Không tìm thấy file ảnh!")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == "":
            flash("Chưa chọn file ảnh!")
            return redirect(request.url)
        
        # Lưu ảnh gốc vào thư mục UPLOAD_IMAGES
        filename = secure_filename(file.filename)
        original_path = os.path.join(UPLOAD_IMAGES, filename)
        file.save(original_path)
        
        # Chuyển ảnh sang tensor
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img_pil = Image.open(original_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Tạo watermark 4-bit ngẫu nhiên
        watermark = torch.randint(0, 2, (1, 4)).float().to(device)
        
        with torch.no_grad():
            watermarked_img = encoder(img_tensor, watermark)
        
        # Lưu ảnh đã nhúng watermark vào thư mục WATERMARKED_DIR
        wm_name = "wm_" + filename
        wm_path = os.path.join(WATERMARKED_DIR, wm_name)
        out_pil = transforms.ToPILImage()(watermarked_img.squeeze(0).cpu())
        out_pil.save(wm_path)
        
        flash("Đã nhúng watermark xong! Ảnh lưu ở data/watermarks/")
        # Hiển thị đường dẫn ảnh, chuyển đổi đường dẫn để dùng trong url_for('static')
        wm_relative_path = wm_path.replace(BASE_DIR, "").lstrip("\\/").replace("\\", "/")
        return render_template("watermark.html", watermarked_image=wm_relative_path)
    
    return render_template("watermark.html")

##############################################
# 5. Trang Face Swap: Hoán đổi 2 ảnh (gốc hoặc watermark)
##############################################
@app.route("/faceswap", methods=["GET", "POST"])
def faceswap_page():
    if request.method == "POST":
        if 'face_a' not in request.files or 'face_b' not in request.files:
            flash("Cần chọn đủ 2 ảnh!")
            return redirect(request.url)
        
        file_a = request.files['face_a']
        file_b = request.files['face_b']
        if file_a.filename == "" or file_b.filename == "":
            flash("Chưa chọn đủ 2 file ảnh!")
            return redirect(request.url)
        
        filename_a = secure_filename(file_a.filename)
        filename_b = secure_filename(file_b.filename)
        
        # Lưu ảnh tải lên vào UPLOAD_IMAGES
        path_a = os.path.join(UPLOAD_IMAGES, filename_a)
        path_b = os.path.join(UPLOAD_IMAGES, filename_b)
        file_a.save(path_a)
        file_b.save(path_b)
        
        # Load ảnh dưới dạng tensor
        face_a = load_image(path_a).to(device)
        face_b = load_image(path_b).to(device)
        
        with torch.no_grad():
            rec_a, rec_b, swapped_a, swapped_b = face_swap_model(face_a, face_b)
        
        # Đặt tên file kết quả và lưu vào RESULT_DIR
        rec_a_name = "rec_a_" + filename_a
        rec_b_name = "rec_b_" + filename_b
        swapped_a_name = "swapped_a_" + filename_a
        swapped_b_name = "swapped_b_" + filename_b
        
        rec_a_path = os.path.join(RESULT_DIR, rec_a_name)
        rec_b_path = os.path.join(RESULT_DIR, rec_b_name)
        swapped_a_path = os.path.join(RESULT_DIR, swapped_a_name)
        swapped_b_path = os.path.join(RESULT_DIR, swapped_b_name)
        
        save_image(rec_a, rec_a_path)
        save_image(rec_b, rec_b_path)
        save_image(swapped_a, swapped_a_path)
        save_image(swapped_b, swapped_b_path)
        
        flash("Face Swap xong! (Ảnh có watermark sẽ phá hoại quá trình swap nếu watermark đã được nhúng đầy đủ.)")
        return render_template("face_swap_result.html",
                               rec_a=url_for('static', filename=f'results/{rec_a_name}'),
                               rec_b=url_for('static', filename=f'results/{rec_b_name}'),
                               swapped_a=url_for('static', filename=f'results/{swapped_a_name}'),
                               swapped_b=url_for('static', filename=f'results/{swapped_b_name}'))
    
    return render_template("face_swap.html")

if __name__ == '__main__':
    app.run(debug=True)
