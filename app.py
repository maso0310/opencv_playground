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


# ===== 程式實作區 =====

# 用於儲存各 session 的變數
code_sessions = {}

# 範例圖片路徑
SAMPLE_IMAGE_PATH = os.path.join(BASE_DIR, 'static', 'sample', 'plant_sample.jpg')


@real_app.route('/vi_lab')
def vi_lab():
    """程式實作區頁面"""
    return render_template('vi_lab.html', vi_indices=VEGETATION_INDICES)


@real_app.route('/get_sample_image')
def get_sample_image():
    """取得範例圖片"""
    if os.path.exists(SAMPLE_IMAGE_PATH):
        img = cv2.imread(SAMPLE_IMAGE_PATH)
        if img is not None:
            # 限制大小
            max_size = 500
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{image_to_base64(img)}',
                'shape': list(img.shape)
            })
    return jsonify({'error': '範例圖片不存在'}), 404


@real_app.route('/execute_code', methods=['POST'])
def execute_code():
    """執行學生程式碼"""
    data = request.get_json()
    code = data.get('code', '')
    session_id = data.get('session_id', 'default')
    cell_id = data.get('cell_id', 0)

    # 初始化 session
    if session_id not in code_sessions:
        code_sessions[session_id] = {
            'variables': {},
            'outputs': []
        }

    session = code_sessions[session_id]

    # 讀取範例圖片作為 img 變數
    sample_img = None
    if os.path.exists(SAMPLE_IMAGE_PATH):
        sample_img = cv2.imread(SAMPLE_IMAGE_PATH)
        if sample_img is not None:
            max_size = 500
            h, w = sample_img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                sample_img = cv2.resize(sample_img, None, fx=scale, fy=scale)

    # 準備輸出捕捉
    output_lines = []

    def capture_print(*args, **kwargs):
        output_lines.append(' '.join(str(a) for a in args))

    # 建立執行環境 (使用預設 builtins，不做限制)
    exec_globals = {
        'cv2': cv2,
        'np': np,
        'numpy': np,
        'print': capture_print,  # 覆蓋 print 以捕捉輸出
    }

    # 加入之前 session 的變數
    exec_globals.update(session['variables'])

    # 如果沒有 img 變數，加入範例圖片
    if 'img' not in exec_globals and sample_img is not None:
        exec_globals['img'] = sample_img.copy()

    try:
        # 執行程式碼
        exec(code, exec_globals)

        # 儲存變數到 session (排除模組和內建)
        for key, value in exec_globals.items():
            if not key.startswith('_') and key not in ['cv2', 'np', 'numpy']:
                if isinstance(value, (np.ndarray, int, float, str, list, dict, tuple)):
                    session['variables'][key] = value

        # 準備回傳的圖片結果
        images = {}
        variables_info = {}

        for key, value in session['variables'].items():
            if isinstance(value, np.ndarray):
                if value.ndim == 2 or (value.ndim == 3 and value.shape[2] in [1, 3, 4]):
                    # 是圖片
                    try:
                        if value.ndim == 2:
                            # 灰階圖，正規化顯示
                            display_img = value.copy()
                            if display_img.dtype != np.uint8:
                                if display_img.max() > 1 or display_img.min() < 0:
                                    display_img = cv2.normalize(display_img, None, 0, 255, cv2.NORM_MINMAX)
                                else:
                                    display_img = (display_img * 255).clip(0, 255)
                                display_img = display_img.astype(np.uint8)
                            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                        else:
                            display_img = value.copy()
                            if display_img.dtype != np.uint8:
                                display_img = display_img.astype(np.uint8)

                        images[key] = f'data:image/png;base64,{image_to_base64(display_img)}'
                        variables_info[key] = f'ndarray {value.shape} {value.dtype}'
                    except:
                        variables_info[key] = f'ndarray {value.shape} {value.dtype}'
                else:
                    variables_info[key] = f'ndarray {value.shape} {value.dtype}'
            elif isinstance(value, (int, float)):
                variables_info[key] = f'{type(value).__name__}: {value}'
            elif isinstance(value, str):
                variables_info[key] = f'str: "{value[:50]}..."' if len(value) > 50 else f'str: "{value}"'
            elif isinstance(value, (list, tuple)):
                variables_info[key] = f'{type(value).__name__} len={len(value)}'
            elif isinstance(value, dict):
                variables_info[key] = f'dict keys={list(value.keys())[:5]}'

        return jsonify({
            'success': True,
            'output': '\n'.join(output_lines),
            'images': images,
            'variables': variables_info
        })

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_msg
        })


@real_app.route('/reset_session', methods=['POST'])
def reset_session():
    """重置 session"""
    data = request.get_json()
    session_id = data.get('session_id', 'default')

    if session_id in code_sessions:
        del code_sessions[session_id]

    return jsonify({'success': True, 'message': 'Session 已重置'})


@real_app.route('/get_session_variables', methods=['POST'])
def get_session_variables():
    """取得 session 中的變數"""
    data = request.get_json()
    session_id = data.get('session_id', 'default')

    if session_id not in code_sessions:
        return jsonify({'variables': {}})

    session = code_sessions[session_id]
    variables_info = {}

    for key, value in session['variables'].items():
        if isinstance(value, np.ndarray):
            variables_info[key] = f'ndarray {value.shape} {value.dtype}'
        else:
            variables_info[key] = f'{type(value).__name__}'

    return jsonify({'variables': variables_info})


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
