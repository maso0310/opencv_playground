# -*- coding: utf-8 -*-
"""
OpenCV 教學平台
讓學員上傳圖片並體驗各種 OpenCV 效果

部署路徑: /opencv_playground
使用 Gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 app:app
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import cv2
import numpy as np
import os
import uuid
import base64
from io import BytesIO

from cv_functions import get_all_effects, process_image
from numpy_functions import get_all_numpy_effects, process_numpy_operation
from vi_functions import (get_all_vi_info, process_vi_step, process_vi_full,
                          VEGETATION_INDICES)

# 建立 Flask 應用
real_app = Flask(__name__)
real_app.secret_key = 'opencv_playground_2024'

# 路徑設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
SUBMISSIONS_FOLDER = os.path.join(BASE_DIR, 'static', 'submissions')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_CODE_EXTENSIONS = {'py'}

real_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
real_app.config['SUBMISSIONS_FOLDER'] = SUBMISSIONS_FOLDER
real_app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUBMISSIONS_FOLDER, exist_ok=True)


def allowed_file(filename):
    """檢查檔案類型"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img):
    """將 OpenCV 影像轉換為 Base64"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


# ===== 路由 =====

@real_app.route('/')
def index():
    """首頁"""
    effects = get_all_effects()

    # 按類別分組
    categories = {}
    for effect_id, info in effects.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({'id': effect_id, 'name': info['name']})

    return render_template('index.html', categories=categories)


@real_app.route('/upload', methods=['POST'])
def upload():
    """上傳圖片"""
    if 'file' not in request.files:
        return jsonify({'error': '沒有選擇檔案'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '沒有選擇檔案'}), 400

    if file and allowed_file(file.filename):
        # 產生唯一檔名
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # 確保資料夾存在
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # 儲存檔案
        file.save(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'url': url_for('static', filename=f'uploads/{filename}')
        })

    return jsonify({'error': '不支援的檔案格式'}), 400


@real_app.route('/process', methods=['POST'])
def process():
    """處理影像"""
    data = request.get_json()

    filename = data.get('filename')
    effect = data.get('effect', 'original')
    params = data.get('params', {})

    if not filename:
        return jsonify({'error': '缺少圖片'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': '圖片不存在'}), 404

    # 讀取圖片
    img = cv2.imread(filepath)

    if img is None:
        return jsonify({'error': '無法讀取圖片'}), 400

    # 限制圖片大小以加快處理速度
    max_size = 800
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # 處理影像
    try:
        result_img, code = process_image(img, effect, params)

        # 轉換為 Base64
        result_base64 = image_to_base64(result_img)

        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{result_base64}',
            'code': code
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@real_app.route('/effects')
def effects():
    """取得所有效果列表"""
    return jsonify(get_all_effects())


@real_app.route('/numpy_effects')
def numpy_effects():
    """取得所有 NumPy 效果列表"""
    return jsonify(get_all_numpy_effects())


@real_app.route('/process_numpy', methods=['POST'])
def process_numpy():
    """處理 NumPy 操作"""
    data = request.get_json()

    effect = data.get('effect', 'array_create')
    params = data.get('params', {})
    source = data.get('source', 'sample')  # 'sample' or 'image'
    filename = data.get('filename')

    img = None

    # 如果需要圖片
    if source == 'image' and filename:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                # 限制圖片大小
                max_size = 400
                h, w = img.shape[:2]
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale)

    try:
        input_data, output_data, code, extra_info = process_numpy_operation(effect, params, img)

        # 準備回傳資料
        response = {
            'success': True,
            'code': code,
            'extra_info': extra_info
        }

        # 處理輸入資料
        if input_data is not None:
            if isinstance(input_data, list):
                # 多個輸入陣列
                response['input_arrays'] = []
                for arr in input_data:
                    if isinstance(arr, np.ndarray):
                        if arr.ndim == 3:  # 圖片
                            response['input_image'] = f'data:image/png;base64,{image_to_base64(arr)}'
                        else:
                            response['input_arrays'].append({
                                'data': arr.tolist(),
                                'shape': arr.shape,
                                'dtype': str(arr.dtype)
                            })
            elif isinstance(input_data, np.ndarray):
                if input_data.ndim == 3:  # 圖片
                    response['input_image'] = f'data:image/png;base64,{image_to_base64(input_data)}'
                else:
                    response['input_array'] = {
                        'data': input_data.tolist(),
                        'shape': input_data.shape,
                        'dtype': str(input_data.dtype)
                    }

        # 處理輸出資料
        if isinstance(output_data, np.ndarray):
            if output_data.ndim == 3:  # 圖片
                response['output_image'] = f'data:image/png;base64,{image_to_base64(output_data)}'
            else:
                response['output_array'] = {
                    'data': output_data.tolist(),
                    'shape': output_data.shape,
                    'dtype': str(output_data.dtype)
                }
        elif isinstance(output_data, (int, float)):
            response['output_scalar'] = output_data
        elif isinstance(output_data, list):
            response['output_list'] = output_data
        elif isinstance(output_data, dict):
            response['output_dict'] = output_data

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@real_app.route('/playground')
def playground():
    """操作頁面"""
    # OpenCV 效果
    effects = get_all_effects()
    categories = {}
    for effect_id, info in effects.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({'id': effect_id, 'name': info['name']})

    # NumPy 效果
    numpy_effects = get_all_numpy_effects()
    numpy_categories = {}
    for effect_id, info in numpy_effects.items():
        cat = info['category']
        if cat not in numpy_categories:
            numpy_categories[cat] = []
        numpy_categories[cat].append({
            'id': effect_id,
            'name': info['name'],
            'requires_image': info.get('requires_image', False)
        })

    return render_template('playground.html',
                           categories=categories,
                           numpy_categories=numpy_categories)


# ===== 植生指標實作 =====

@real_app.route('/vi_lab')
def vi_lab():
    """植生指標實作頁面"""
    return render_template('vi_lab.html', vi_indices=VEGETATION_INDICES)


@real_app.route('/vi_info')
def vi_info():
    """取得植生指標資訊"""
    return jsonify(get_all_vi_info())


@real_app.route('/process_vi_step', methods=['POST'])
def process_vi_step_route():
    """分步驟處理植生指標"""
    data = request.get_json()

    filename = data.get('filename')
    vi_type = data.get('vi_type', 'ExG')
    step = data.get('step', 1)

    if not filename:
        return jsonify({'error': '缺少圖片'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': '圖片不存在'}), 404

    img = cv2.imread(filepath)

    if img is None:
        return jsonify({'error': '無法讀取圖片'}), 400

    # 限制圖片大小
    max_size = 600
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    try:
        result = process_vi_step(img, vi_type, step)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@real_app.route('/process_vi_full', methods=['POST'])
def process_vi_full_route():
    """完整處理植生指標"""
    data = request.get_json()

    filename = data.get('filename')
    vi_type = data.get('vi_type', 'ExG')

    if not filename:
        return jsonify({'error': '缺少圖片'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': '圖片不存在'}), 404

    img = cv2.imread(filepath)

    if img is None:
        return jsonify({'error': '無法讀取圖片'}), 400

    # 限制圖片大小
    max_size = 600
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    try:
        result = process_vi_full(img, vi_type)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ===== 作業上傳 =====

@real_app.route('/submissions')
def submissions():
    """作業上傳與展示頁面"""
    # 讀取已上傳的作業
    submissions_data = {}
    for group_num in range(1, 7):
        group_folder = os.path.join(SUBMISSIONS_FOLDER, f'group_{group_num}')
        if os.path.exists(group_folder):
            files = os.listdir(group_folder)
            submissions_data[group_num] = {
                'has_submission': len(files) > 0,
                'files': files
            }
        else:
            submissions_data[group_num] = {
                'has_submission': False,
                'files': []
            }

    return render_template('submissions.html',
                           vi_indices=VEGETATION_INDICES,
                           submissions=submissions_data)


@real_app.route('/submit_assignment', methods=['POST'])
def submit_assignment():
    """上傳作業"""
    group_num = request.form.get('group_num')
    vi_type = request.form.get('vi_type')

    if not group_num or not vi_type:
        return jsonify({'error': '缺少組別或指標類型'}), 400

    # 建立組別資料夾
    group_folder = os.path.join(SUBMISSIONS_FOLDER, f'group_{group_num}')
    os.makedirs(group_folder, exist_ok=True)

    saved_files = []

    # 處理上傳的檔案
    file_types = ['original', 'vi_map', 'mask', 'result', 'histogram', 'code']

    for file_type in file_types:
        if file_type in request.files:
            file = request.files[file_type]
            if file and file.filename:
                # 產生檔名
                ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'png'
                filename = f'{vi_type}_{file_type}.{ext}'
                filepath = os.path.join(group_folder, filename)
                file.save(filepath)
                saved_files.append(filename)

    return jsonify({
        'success': True,
        'message': f'第 {group_num} 組作業上傳成功',
        'files': saved_files
    })


@real_app.route('/get_submission/<int:group_num>')
def get_submission(group_num):
    """取得特定組別的作業內容"""
    group_folder = os.path.join(SUBMISSIONS_FOLDER, f'group_{group_num}')

    if not os.path.exists(group_folder):
        return jsonify({'error': '該組別尚未上傳作業'}), 404

    files = os.listdir(group_folder)
    submission_data = {
        'group_num': group_num,
        'files': {}
    }

    for filename in files:
        filepath = os.path.join(group_folder, filename)

        if filename.endswith('.py'):
            # 讀取程式碼
            with open(filepath, 'r', encoding='utf-8') as f:
                submission_data['files'][filename] = {
                    'type': 'code',
                    'content': f.read()
                }
        else:
            # 圖片轉 Base64
            submission_data['files'][filename] = {
                'type': 'image',
                'url': url_for('static', filename=f'submissions/group_{group_num}/{filename}')
            }

    return jsonify(submission_data)


# ===== DispatcherMiddleware 設定 =====

app = DispatcherMiddleware(
    lambda environ, start_response: (
        start_response('404 Not Found', [('Content-Type', 'text/plain')]) or [b'Not Found']
    ),
    {
        "/opencv_playground": real_app
    }
)


# ===== 啟動 =====

if __name__ == '__main__':
    # 確保上傳資料夾存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    print("=" * 50)
    print("OpenCV 教學平台")
    print("=" * 50)
    print("\n本地測試: http://localhost:5000/opencv_playground/")
    print("VPS 部署: gunicorn -w 4 -b 0.0.0.0:5000 app:app")
    print("=" * 50 + "\n")

    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 5000, app, use_reloader=True, use_debugger=True)
