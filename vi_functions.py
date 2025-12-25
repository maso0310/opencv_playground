# -*- coding: utf-8 -*-
"""
植生指標 (Vegetation Index) 處理模組
提供各種植生指標的計算與處理函數
"""

import cv2
import numpy as np
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 設定 matplotlib 中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===== 植生指標定義 =====

VEGETATION_INDICES = {
    'ExG': {
        'name': 'ExG (Excess Green)',
        'formula': '2G - R - B',
        'description': '最常用的植生指標，強調綠色區域。數值越高表示越可能是植被。',
        'vegetation_high': True  # True 表示高值為植被
    },
    'ExGR': {
        'name': 'ExGR (Excess Green minus Red)',
        'formula': '3G - 2.4R - B',
        'description': '結合 ExG 和 ExR 的優點，對紅色區域有更好的抑制效果。',
        'vegetation_high': True
    },
    'GLI': {
        'name': 'GLI (Green Leaf Index)',
        'formula': '(2G - R - B) / (2G + R + B)',
        'description': '正規化的綠葉指標，數值範圍 -1 到 1。正值表示植被。',
        'vegetation_high': True
    },
    'VARI': {
        'name': 'VARI (Visible Atmospherically Resistant Index)',
        'formula': '(G - R) / (G + R - B)',
        'description': '對大氣影響有較好的抗性，適合戶外環境。',
        'vegetation_high': True
    },
    'NGRDI': {
        'name': 'NGRDI (Normalized Green-Red Difference Index)',
        'formula': '(G - R) / (G + R)',
        'description': '最簡單的正規化指標，只使用 G 和 R 通道。數值範圍 -1 到 1。',
        'vegetation_high': True
    },
    'CIVE': {
        'name': 'CIVE (Color Index of Vegetation Extraction)',
        'formula': '0.441R - 0.811G + 0.385B + 18.787',
        'description': '使用經驗係數的指標。注意：數值越低表示越可能是植被。',
        'vegetation_high': False  # False 表示低值為植被
    }
}


def get_all_vi_info():
    """取得所有植生指標資訊"""
    return VEGETATION_INDICES


def calculate_vi(img, vi_type):
    """
    計算植生指標

    Args:
        img: BGR 格式的影像 (numpy array)
        vi_type: 植生指標類型 ('ExG', 'ExGR', 'GLI', 'VARI', 'NGRDI', 'CIVE')

    Returns:
        vi_map: 植生指標圖 (float32)
        code: 對應的 Python 程式碼
    """
    # 分離通道
    B, G, R = cv2.split(img)

    # 轉為浮點數
    B = B.astype(np.float32)
    G = G.astype(np.float32)
    R = R.astype(np.float32)

    if vi_type == 'ExG':
        # 正規化到 0-1
        B_norm = B / 255.0
        G_norm = G / 255.0
        R_norm = R / 255.0
        vi_map = 2 * G_norm - R_norm - B_norm
        code = """# ExG (Excess Green) 計算
B, G, R = cv2.split(img)

# 正規化到 0-1
B = B.astype(np.float32) / 255.0
G = G.astype(np.float32) / 255.0
R = R.astype(np.float32) / 255.0

# 計算 ExG
ExG = 2 * G - R - B"""

    elif vi_type == 'ExGR':
        B_norm = B / 255.0
        G_norm = G / 255.0
        R_norm = R / 255.0
        vi_map = 3 * G_norm - 2.4 * R_norm - B_norm
        code = """# ExGR (Excess Green minus Red) 計算
B, G, R = cv2.split(img)

# 正規化到 0-1
B = B.astype(np.float32) / 255.0
G = G.astype(np.float32) / 255.0
R = R.astype(np.float32) / 255.0

# 計算 ExGR = ExG - ExR = (2G - R - B) - (1.4R - G)
ExGR = 3 * G - 2.4 * R - B"""

    elif vi_type == 'GLI':
        B_norm = B / 255.0
        G_norm = G / 255.0
        R_norm = R / 255.0
        numerator = 2 * G_norm - R_norm - B_norm
        denominator = 2 * G_norm + R_norm + B_norm + 1e-10
        vi_map = numerator / denominator
        code = """# GLI (Green Leaf Index) 計算
B, G, R = cv2.split(img)

# 正規化到 0-1
B = B.astype(np.float32) / 255.0
G = G.astype(np.float32) / 255.0
R = R.astype(np.float32) / 255.0

# 計算 GLI (加小數避免除以零)
numerator = 2 * G - R - B
denominator = 2 * G + R + B + 1e-10
GLI = numerator / denominator"""

    elif vi_type == 'VARI':
        B_norm = B / 255.0
        G_norm = G / 255.0
        R_norm = R / 255.0
        numerator = G_norm - R_norm
        denominator = G_norm + R_norm - B_norm + 1e-10
        vi_map = numerator / denominator
        vi_map = np.clip(vi_map, -1, 1)  # 限制範圍
        code = """# VARI (Visible Atmospherically Resistant Index) 計算
B, G, R = cv2.split(img)

# 正規化到 0-1
B = B.astype(np.float32) / 255.0
G = G.astype(np.float32) / 255.0
R = R.astype(np.float32) / 255.0

# 計算 VARI
numerator = G - R
denominator = G + R - B + 1e-10
VARI = numerator / denominator
VARI = np.clip(VARI, -1, 1)  # 限制極端值"""

    elif vi_type == 'NGRDI':
        B_norm = B / 255.0
        G_norm = G / 255.0
        R_norm = R / 255.0
        numerator = G_norm - R_norm
        denominator = G_norm + R_norm + 1e-10
        vi_map = numerator / denominator
        code = """# NGRDI (Normalized Green-Red Difference Index) 計算
B, G, R = cv2.split(img)

# 正規化到 0-1
B = B.astype(np.float32) / 255.0
G = G.astype(np.float32) / 255.0
R = R.astype(np.float32) / 255.0

# 計算 NGRDI
numerator = G - R
denominator = G + R + 1e-10
NGRDI = numerator / denominator"""

    elif vi_type == 'CIVE':
        # CIVE 使用原始 0-255 值
        vi_map = 0.441 * R - 0.811 * G + 0.385 * B + 18.787
        code = """# CIVE (Color Index of Vegetation Extraction) 計算
B, G, R = cv2.split(img)

# 轉為浮點數 (CIVE 使用原始 0-255 值)
B = B.astype(np.float32)
G = G.astype(np.float32)
R = R.astype(np.float32)

# 計算 CIVE (注意：低值表示植被)
CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.787"""

    else:
        raise ValueError(f"未知的植生指標類型: {vi_type}")

    return vi_map, code


def normalize_vi(vi_map):
    """
    將植生指標正規化到 0-255

    Args:
        vi_map: 植生指標圖 (float32)

    Returns:
        vi_normalized: 正規化後的圖 (uint8)
        code: 對應的 Python 程式碼
    """
    vi_normalized = cv2.normalize(vi_map, None, 0, 255, cv2.NORM_MINMAX)
    vi_uint8 = vi_normalized.astype(np.uint8)

    code = """# 正規化到 0-255
vi_normalized = cv2.normalize(vi_map, None, 0, 255, cv2.NORM_MINMAX)
vi_uint8 = vi_normalized.astype(np.uint8)"""

    return vi_uint8, code


def otsu_threshold(vi_uint8, vegetation_high=True):
    """
    使用 Otsu 方法進行二值化

    Args:
        vi_uint8: 正規化後的植生指標圖 (uint8)
        vegetation_high: True 表示高值為植被，False 表示低值為植被

    Returns:
        mask: 二值化遮罩 (白色=植被)
        threshold: Otsu 計算的閾值
        code: 對應的 Python 程式碼
    """
    threshold, mask = cv2.threshold(vi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 如果低值為植被，需要反轉遮罩
    if not vegetation_high:
        mask = cv2.bitwise_not(mask)
        code = f"""# Otsu 二值化
threshold, mask = cv2.threshold(vi_uint8, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 注意：此指標低值表示植被，需要反轉遮罩
mask = cv2.bitwise_not(mask)

print(f"Otsu 閾值: {{threshold}}")"""
    else:
        code = f"""# Otsu 二值化
threshold, mask = cv2.threshold(vi_uint8, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Otsu 閾值: {{threshold}}")"""

    return mask, threshold, code


def apply_mask(img, mask):
    """
    使用遮罩切割圖片

    Args:
        img: 原始 BGR 影像
        mask: 二值化遮罩

    Returns:
        result: 切割後的影像 (植被 + 黑色背景)
        code: 對應的 Python 程式碼
    """
    result = cv2.bitwise_and(img, img, mask=mask)

    code = """# 使用遮罩切割圖片
result = cv2.bitwise_and(img, img, mask=mask)

# 結果：只保留植被區域，背景為黑色"""

    return result, code


def create_histogram(vi_map, vi_type, threshold=None):
    """
    建立植生指標的直方圖

    Args:
        vi_map: 植生指標圖 (float32)
        vi_type: 植生指標類型
        threshold: Otsu 閾值（正規化後的值）

    Returns:
        hist_base64: 直方圖的 Base64 編碼
        code: 對應的 Python 程式碼
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # 計算直方圖
    data = vi_map.flatten()

    # 繪製直方圖
    n, bins, patches = ax.hist(data, bins=50, color='green', alpha=0.7, edgecolor='darkgreen')

    # 標題和標籤
    vi_info = VEGETATION_INDICES.get(vi_type, {})
    ax.set_title(f'{vi_type} 數值分布直方圖', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{vi_type} 數值', fontsize=12)
    ax.set_ylabel('像素數量', fontsize=12)

    # 添加統計資訊
    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)

    stats_text = f'平均值: {mean_val:.4f}\n標準差: {std_val:.4f}\n範圍: [{min_val:.4f}, {max_val:.4f}]'
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 添加閾值線
    if threshold is not None:
        # 將閾值轉換回原始範圍
        vi_min = np.min(data)
        vi_max = np.max(data)
        thresh_original = vi_min + (threshold / 255.0) * (vi_max - vi_min)
        ax.axvline(x=thresh_original, color='red', linestyle='--', linewidth=2,
                   label=f'Otsu 閾值: {thresh_original:.4f}')
        ax.legend(loc='upper left')

    plt.tight_layout()

    # 轉換為 Base64
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    hist_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    code = f"""# 繪製 {vi_type} 直方圖
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.hist(vi_map.flatten(), bins=50, color='green', alpha=0.7)
plt.title('{vi_type} 數值分布直方圖')
plt.xlabel('{vi_type} 數值')
plt.ylabel('像素數量')
plt.savefig('{vi_type}_histogram.png')
plt.show()"""

    return hist_base64, code


def image_to_base64(img):
    """將 OpenCV 影像轉換為 Base64"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def process_vi_step(img, vi_type, step, session_data=None):
    """
    分步驟處理植生指標

    Args:
        img: 原始 BGR 影像
        vi_type: 植生指標類型
        step: 處理步驟 (1-7)
        session_data: 前面步驟的資料（用於延續處理）

    Returns:
        result: 包含處理結果的字典
    """
    vi_info = VEGETATION_INDICES.get(vi_type, {})
    vegetation_high = vi_info.get('vegetation_high', True)

    result = {
        'step': step,
        'vi_type': vi_type,
        'success': True
    }

    if step == 1:
        # 步驟 1: 通道拆解
        B, G, R = cv2.split(img)

        # 建立通道視覺化
        zeros = np.zeros_like(B)
        B_img = cv2.merge([B, zeros, zeros])
        G_img = cv2.merge([zeros, G, zeros])
        R_img = cv2.merge([zeros, zeros, R])

        result['B_channel'] = image_to_base64(B_img)
        result['G_channel'] = image_to_base64(G_img)
        result['R_channel'] = image_to_base64(R_img)
        result['code'] = """import cv2
import numpy as np

# 讀取圖片
img = cv2.imread('plant.jpg')

# 步驟 1: 通道拆解
B, G, R = cv2.split(img)

print(f"B 通道形狀: {B.shape}")
print(f"G 通道形狀: {G.shape}")
print(f"R 通道形狀: {R.shape}")"""
        result['description'] = '將 BGR 彩色影像拆解為三個獨立的通道'

    elif step == 2:
        # 步驟 2: 正規化
        B, G, R = cv2.split(img)
        B_norm = B.astype(np.float32) / 255.0
        G_norm = G.astype(np.float32) / 255.0
        R_norm = R.astype(np.float32) / 255.0

        # 視覺化正規化後的通道
        B_vis = (B_norm * 255).astype(np.uint8)
        G_vis = (G_norm * 255).astype(np.uint8)
        R_vis = (R_norm * 255).astype(np.uint8)

        result['B_normalized'] = image_to_base64(B_vis)
        result['G_normalized'] = image_to_base64(G_vis)
        result['R_normalized'] = image_to_base64(R_vis)
        result['stats'] = {
            'B': {'min': float(np.min(B_norm)), 'max': float(np.max(B_norm)), 'mean': float(np.mean(B_norm))},
            'G': {'min': float(np.min(G_norm)), 'max': float(np.max(G_norm)), 'mean': float(np.mean(G_norm))},
            'R': {'min': float(np.min(R_norm)), 'max': float(np.max(R_norm)), 'mean': float(np.mean(R_norm))}
        }
        result['code'] = """# 步驟 2: 正規化到 0-1 範圍
B = B.astype(np.float32) / 255.0
G = G.astype(np.float32) / 255.0
R = R.astype(np.float32) / 255.0

print(f"正規化後範圍: 0.0 ~ 1.0")
print(f"G 通道平均值: {np.mean(G):.4f}")"""
        result['description'] = '將像素值從 0-255 正規化到 0-1 範圍，方便進行數學運算'

    elif step == 3:
        # 步驟 3: 計算植生指標
        vi_map, code = calculate_vi(img, vi_type)

        # 正規化視覺化
        vi_vis = cv2.normalize(vi_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vi_color = cv2.applyColorMap(vi_vis, cv2.COLORMAP_JET)

        result['vi_map'] = image_to_base64(vi_vis)
        result['vi_map_color'] = image_to_base64(vi_color)
        result['stats'] = {
            'min': float(np.min(vi_map)),
            'max': float(np.max(vi_map)),
            'mean': float(np.mean(vi_map)),
            'std': float(np.std(vi_map))
        }
        result['code'] = code
        result['description'] = f'計算 {vi_type} 植生指標: {vi_info.get("formula", "")}'

    elif step == 4:
        # 步驟 4: Otsu 二值化閾值計算
        vi_map, _ = calculate_vi(img, vi_type)
        vi_uint8, _ = normalize_vi(vi_map)
        _, threshold, code = otsu_threshold(vi_uint8, vegetation_high)

        # 顯示閾值在直方圖上
        result['threshold'] = float(threshold)
        result['code'] = code
        result['description'] = f'使用 Otsu 方法自動計算最佳閾值，結果為 {threshold:.1f}'

    elif step == 5:
        # 步驟 5: 產生二值化遮罩
        vi_map, _ = calculate_vi(img, vi_type)
        vi_uint8, _ = normalize_vi(vi_map)
        mask, threshold, code = otsu_threshold(vi_uint8, vegetation_high)

        result['mask'] = image_to_base64(mask)
        result['threshold'] = float(threshold)
        result['pixel_count'] = {
            'vegetation': int(np.sum(mask == 255)),
            'background': int(np.sum(mask == 0)),
            'total': int(mask.size)
        }
        result['coverage'] = float(np.sum(mask == 255) / mask.size * 100)
        result['code'] = code + """

# 統計植被覆蓋率
vegetation_pixels = np.sum(mask == 255)
total_pixels = mask.size
coverage = vegetation_pixels / total_pixels * 100
print(f"植被覆蓋率: {coverage:.2f}%")"""
        result['description'] = '根據 Otsu 閾值將植生指標圖轉換為二值化遮罩'

    elif step == 6:
        # 步驟 6: 遮罩切割
        vi_map, _ = calculate_vi(img, vi_type)
        vi_uint8, _ = normalize_vi(vi_map)
        mask, _, _ = otsu_threshold(vi_uint8, vegetation_high)
        segmented, code = apply_mask(img, mask)

        result['segmented'] = image_to_base64(segmented)
        result['code'] = code
        result['description'] = '使用遮罩提取植被區域，非植被區域顯示為黑色背景'

    elif step == 7:
        # 步驟 7: 產生直方圖
        vi_map, _ = calculate_vi(img, vi_type)
        vi_uint8, _ = normalize_vi(vi_map)
        _, threshold, _ = otsu_threshold(vi_uint8, vegetation_high)
        hist_base64, code = create_histogram(vi_map, vi_type, threshold)

        result['histogram'] = hist_base64
        result['code'] = code
        result['description'] = f'繪製 {vi_type} 數值分布直方圖，並標示 Otsu 閾值位置'

    else:
        result['success'] = False
        result['error'] = f'未知的步驟: {step}'

    return result


def process_vi_full(img, vi_type):
    """
    完整處理植生指標（一次執行所有步驟）

    Args:
        img: 原始 BGR 影像
        vi_type: 植生指標類型

    Returns:
        result: 包含所有處理結果的字典
    """
    vi_info = VEGETATION_INDICES.get(vi_type, {})
    vegetation_high = vi_info.get('vegetation_high', True)

    # 計算植生指標
    vi_map, vi_code = calculate_vi(img, vi_type)

    # 正規化
    vi_uint8, norm_code = normalize_vi(vi_map)

    # Otsu 二值化
    mask, threshold, otsu_code = otsu_threshold(vi_uint8, vegetation_high)

    # 遮罩切割
    segmented, mask_code = apply_mask(img, mask)

    # 產生直方圖
    hist_base64, hist_code = create_histogram(vi_map, vi_type, threshold)

    # 視覺化
    vi_vis = cv2.normalize(vi_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vi_color = cv2.applyColorMap(vi_vis, cv2.COLORMAP_JET)

    # 組合完整程式碼
    full_code = f"""import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
img = cv2.imread('plant.jpg')

# ===== 步驟 1: 通道拆解 =====
B, G, R = cv2.split(img)

# ===== 步驟 2: 正規化 =====
B = B.astype(np.float32) / 255.0
G = G.astype(np.float32) / 255.0
R = R.astype(np.float32) / 255.0

# ===== 步驟 3: 計算 {vi_type} =====
{vi_code.split('# 計算')[1] if '# 計算' in vi_code else vi_code}

# ===== 步驟 4 & 5: Otsu 二值化 =====
vi_normalized = cv2.normalize({vi_type}, None, 0, 255, cv2.NORM_MINMAX)
vi_uint8 = vi_normalized.astype(np.uint8)
threshold, mask = cv2.threshold(vi_uint8, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
{"mask = cv2.bitwise_not(mask)  # 低值為植被，需反轉" if not vegetation_high else ""}
print(f"Otsu 閾值: {{threshold}}")

# ===== 步驟 6: 遮罩切割 =====
result = cv2.bitwise_and(img, img, mask=mask)

# 計算植被覆蓋率
coverage = np.sum(mask == 255) / mask.size * 100
print(f"植被覆蓋率: {{coverage:.2f}}%")

# ===== 步驟 7: 繪製直方圖 =====
plt.figure(figsize=(10, 4))
plt.hist({vi_type}.flatten(), bins=50, color='green', alpha=0.7)
plt.title('{vi_type} 數值分布直方圖')
plt.xlabel('{vi_type} 數值')
plt.ylabel('像素數量')
plt.savefig('{vi_type}_histogram.png')

# 儲存結果
cv2.imwrite('result_{vi_type}.png', result)
cv2.imwrite('{vi_type}_mask.png', mask)
"""

    return {
        'success': True,
        'vi_type': vi_type,
        'vi_info': vi_info,
        'original': image_to_base64(img),
        'vi_map': image_to_base64(vi_vis),
        'vi_map_color': image_to_base64(vi_color),
        'mask': image_to_base64(mask),
        'segmented': image_to_base64(segmented),
        'histogram': hist_base64,
        'threshold': float(threshold),
        'stats': {
            'vi_min': float(np.min(vi_map)),
            'vi_max': float(np.max(vi_map)),
            'vi_mean': float(np.mean(vi_map)),
            'vi_std': float(np.std(vi_map)),
            'vegetation_pixels': int(np.sum(mask == 255)),
            'total_pixels': int(mask.size),
            'coverage': float(np.sum(mask == 255) / mask.size * 100)
        },
        'code': full_code
    }
