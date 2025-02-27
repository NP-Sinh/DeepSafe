from flask import Flask, render_template, request, redirect, url_for
import os
from watermark import embed_watermark, simulate_deepfake_attack, extract_watermark

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Nhúng watermark vào ảnh gốc
    watermarked_image_path = embed_watermark(filepath)
    return render_template('result.html', original=filepath, watermarked=watermarked_image_path)

@app.route('/simulate', methods=['POST'])
def simulate_attack():
    image_path = request.form.get("image_path")
    attacked_image_path = simulate_deepfake_attack(image_path)
    # Giả sử ta có thể trích xuất watermark từ ảnh bị tấn công
    watermark_info = extract_watermark(attacked_image_path)
    return render_template('result.html', original=image_path, watermarked=attacked_image_path, watermark=watermark_info)

if __name__ == '__main__':
    app.run(debug=True)
