# src/app.py
from flask import Flask, render_template, request, redirect, url_for
from face_swap.swap import perform_face_swap
import os

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_FOLDER = os.path.join(BASE_DIR, '..', 'data', 'processed')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/swapface', methods=['POST'])
def swapface():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Gọi hàm thực hiện face swap, hàm này bao gồm cả biến dạng bảo vệ
    result_path = perform_face_swap(filepath)
    
    # Đảm bảo đường dẫn hiển thị trên giao diện phù hợp với static folder nếu cần
    static_path = result_path.split('static/')[-1]
    return render_template('result.html', result_image=static_path)

if __name__ == '__main__':
    app.run(debug=True)
