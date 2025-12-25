# -*- coding: utf-8 -*-
"""
OpenCV 教學平台 - 影像處理函數模組
包含所有 OpenCV 效果的處理函數與程式碼說明
"""

import cv2
import numpy as np
import os

# Haar Cascade 路徑
CASCADE_PATH = os.path.join(os.path.dirname(__file__), 'haarcascades')


def get_all_effects():
    """取得所有可用的效果列表"""
    return {
        'original': {'name': '原始影像', 'category': '基礎'},
        'channels': {'name': '通道拆解 (BGR)', 'category': '基礎'},
        'arithmetic': {'name': '影像運算', 'category': '基礎'},
        'weighted': {'name': '影像加權和', 'category': '基礎'},
        'bitwise': {'name': '逐位元邏輯運算', 'category': '基礎'},
        'mask': {'name': '掩模 (Mask)', 'category': '基礎'},
        'bit_plane': {'name': '位元平面分解', 'category': '基礎'},
        'encrypt': {'name': '影像加密與解密', 'category': '基礎'},
        'color_space': {'name': '色彩空間轉換', 'category': '色彩'},
        'alpha_channel': {'name': '透明圖層通道', 'category': '色彩'},
        'geometric': {'name': '幾何轉換', 'category': '幾何'},
        'threshold': {'name': '二值化處理', 'category': '二值化'},
        'blur_mean': {'name': '均值濾波', 'category': '平滑處理'},
        'blur_box': {'name': '方框濾波', 'category': '平滑處理'},
        'blur_gaussian': {'name': '高斯濾波', 'category': '平滑處理'},
        'blur_median': {'name': '中值濾波', 'category': '平滑處理'},
        'blur_bilateral': {'name': '雙邊濾波', 'category': '平滑處理'},
        'morphology': {'name': '形態學操作', 'category': '形態學'},
        'gradient_sobel': {'name': 'Sobel 梯度', 'category': '梯度'},
        'gradient_scharr': {'name': 'Scharr 梯度', 'category': '梯度'},
        'gradient_laplacian': {'name': 'Laplacian 梯度', 'category': '梯度'},
        'canny': {'name': 'Canny 邊緣檢測', 'category': '邊緣檢測'},
        'pyramid': {'name': '影像金字塔', 'category': '金字塔'},
        'contours': {'name': '影像輪廓', 'category': '輪廓'},
        'histogram': {'name': '長條圖處理', 'category': '長條圖'},
        'fourier': {'name': '傅立葉轉換', 'category': '頻域'},
        'hough_line': {'name': '霍夫直線轉換', 'category': '霍夫'},
        'hough_circle': {'name': '霍夫圓形轉換', 'category': '霍夫'},
        'watershed': {'name': '分水嶺演算法', 'category': '分割'},
        'face_detection': {'name': '人臉識別', 'category': '識別'},
    }


def process_image(img, effect, params=None):
    """
    處理影像並返回結果和程式碼

    Args:
        img: OpenCV 影像 (BGR)
        effect: 效果名稱
        params: 額外參數

    Returns:
        (processed_img, code_str)
    """
    if params is None:
        params = {}

    processors = {
        'original': process_original,
        'channels': process_channels,
        'arithmetic': process_arithmetic,
        'weighted': process_weighted,
        'bitwise': process_bitwise,
        'mask': process_mask,
        'bit_plane': process_bit_plane,
        'encrypt': process_encrypt,
        'color_space': process_color_space,
        'alpha_channel': process_alpha_channel,
        'geometric': process_geometric,
        'threshold': process_threshold,
        'blur_mean': process_blur_mean,
        'blur_box': process_blur_box,
        'blur_gaussian': process_blur_gaussian,
        'blur_median': process_blur_median,
        'blur_bilateral': process_blur_bilateral,
        'morphology': process_morphology,
        'gradient_sobel': process_gradient_sobel,
        'gradient_scharr': process_gradient_scharr,
        'gradient_laplacian': process_gradient_laplacian,
        'canny': process_canny,
        'pyramid': process_pyramid,
        'contours': process_contours,
        'histogram': process_histogram,
        'fourier': process_fourier,
        'hough_line': process_hough_line,
        'hough_circle': process_hough_circle,
        'watershed': process_watershed,
        'face_detection': process_face_detection,
    }

    processor = processors.get(effect, process_original)
    return processor(img, params)


# ===== 基礎處理 =====

def process_original(img, params):
    """原始影像"""
    code = '''import cv2

# 讀取影像
img = cv2.imread('image.jpg')

# 顯示影像
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
    return img, code


def process_channels(img, params):
    """通道拆解"""
    b, g, r = cv2.split(img)

    # 建立彩色顯示的各通道
    zeros = np.zeros_like(b)
    b_img = cv2.merge([b, zeros, zeros])
    g_img = cv2.merge([zeros, g, zeros])
    r_img = cv2.merge([zeros, zeros, r])

    # 橫向拼接
    h, w = img.shape[:2]
    scale = 0.33
    b_small = cv2.resize(b_img, (int(w*scale), int(h*scale)))
    g_small = cv2.resize(g_img, (int(w*scale), int(h*scale)))
    r_small = cv2.resize(r_img, (int(w*scale), int(h*scale)))
    result = np.hstack([b_small, g_small, r_small])

    code = '''import cv2
import numpy as np

# 讀取影像
img = cv2.imread('image.jpg')

# 拆解 BGR 三個通道
b, g, r = cv2.split(img)

# 也可以直接用索引取得
# b = img[:, :, 0]  # Blue 通道
# g = img[:, :, 1]  # Green 通道
# r = img[:, :, 2]  # Red 通道

# 合併通道
merged = cv2.merge([b, g, r])

# 建立單獨顯示各通道的彩色圖
zeros = np.zeros_like(b)
b_img = cv2.merge([b, zeros, zeros])  # 只顯示藍色
g_img = cv2.merge([zeros, g, zeros])  # 只顯示綠色
r_img = cv2.merge([zeros, zeros, r])  # 只顯示紅色'''
    return result, code


def process_arithmetic(img, params):
    """影像運算"""
    # 加亮
    bright = cv2.add(img, np.ones_like(img) * 50)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 影像加法 - 加亮
bright = cv2.add(img, np.ones_like(img) * 50)

# 影像減法 - 變暗
dark = cv2.subtract(img, np.ones_like(img) * 50)

# 注意：cv2.add() 會自動處理溢位 (超過255變成255)
# 而 img + 50 可能會溢位 (超過255會從0重新計算)

# 兩張影像相加
# result = cv2.add(img1, img2)

# 兩張影像相減
# result = cv2.subtract(img1, img2)'''
    return bright, code


def process_weighted(img, params):
    """影像加權和"""
    # 建立一個漸層圖
    h, w = img.shape[:2]
    gradient = np.zeros_like(img)
    for i in range(w):
        gradient[:, i] = [int(255 * i / w)] * 3

    # 加權合成
    alpha = params.get('alpha', 0.7)
    result = cv2.addWeighted(img, alpha, gradient, 1-alpha, 0)

    code = '''import cv2

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# 影像加權和: dst = src1 * alpha + src2 * beta + gamma
# alpha + beta 通常 = 1
alpha = 0.7
beta = 0.3
gamma = 0  # 亮度調整

result = cv2.addWeighted(img1, alpha, img2, beta, gamma)

# 應用場景：
# - 影像融合
# - 浮水印
# - 淡入淡出效果'''
    return result, code


def process_bitwise(img, params):
    """逐位元邏輯運算"""
    h, w = img.shape[:2]

    # 建立一個圓形遮罩
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//3, 255, -1)
    mask_3ch = cv2.merge([mask, mask, mask])

    # AND 運算
    result = cv2.bitwise_and(img, mask_3ch)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 建立遮罩
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (w//2, h//2), 100, 255, -1)

# 逐位元 AND 運算
result_and = cv2.bitwise_and(img, img, mask=mask)

# 逐位元 OR 運算
result_or = cv2.bitwise_or(img1, img2)

# 逐位元 XOR 運算
result_xor = cv2.bitwise_xor(img1, img2)

# 逐位元 NOT 運算 (反轉)
result_not = cv2.bitwise_not(img)'''
    return result, code


def process_mask(img, params):
    """掩模"""
    h, w = img.shape[:2]

    # 建立多邊形遮罩
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[w//4, h//4], [3*w//4, h//4],
                    [3*w//4, 3*h//4], [w//4, 3*h//4]], np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # 套用遮罩
    result = cv2.bitwise_and(img, img, mask=mask)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 建立空白遮罩 (黑色背景)
mask = np.zeros((h, w), dtype=np.uint8)

# 在遮罩上繪製白色區域 (要保留的部分)
# 方形
cv2.rectangle(mask, (50, 50), (200, 200), 255, -1)

# 圓形
cv2.circle(mask, (w//2, h//2), 100, 255, -1)

# 多邊形
pts = np.array([[100, 50], [200, 150], [100, 250]], np.int32)
cv2.fillPoly(mask, [pts], 255)

# 套用遮罩
result = cv2.bitwise_and(img, img, mask=mask)

# 遮罩的應用：
# - 選取特定區域
# - 去除背景
# - 局部處理'''
    return result, code


def process_bit_plane(img, params):
    """位元平面分解"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 取得各位元平面
    planes = []
    for i in range(8):
        plane = (gray >> i) & 1
        plane = plane * 255
        planes.append(plane)

    # 顯示最高位元平面 (包含最多資訊)
    result = cv2.cvtColor(planes[7].astype(np.uint8), cv2.COLOR_GRAY2BGR)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 8位元灰階影像有8個位元平面
# 最低位元 (LSB) 到最高位元 (MSB)

planes = []
for i in range(8):
    # 右移 i 位，然後取最低位
    plane = (img >> i) & 1
    # 放大到 0-255 以便顯示
    plane = plane * 255
    planes.append(plane)

# planes[0] = 最低位元平面 (雜訊較多)
# planes[7] = 最高位元平面 (主要資訊)

# 位元平面的應用：
# - 影像壓縮
# - 浮水印嵌入
# - 資訊隱藏 (LSB 隱寫術)'''
    return result, code


def process_encrypt(img, params):
    """影像加密與解密"""
    # 使用 XOR 加密
    h, w = img.shape[:2]
    np.random.seed(42)  # 固定種子以便解密
    key = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    encrypted = cv2.bitwise_xor(img, key)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 產生隨機金鑰
np.random.seed(42)  # 設定種子以便之後解密
key = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

# 加密 (XOR 運算)
encrypted = cv2.bitwise_xor(img, key)

# 解密 (再次 XOR 同樣的金鑰)
decrypted = cv2.bitwise_xor(encrypted, key)

# XOR 加密的特性：
# A XOR B = C
# C XOR B = A
# 相同的金鑰可以加密和解密'''
    return encrypted, code


# ===== 色彩處理 =====

def process_color_space(img, params):
    """色彩空間轉換"""
    mode = params.get('mode', 'hsv')

    if mode == 'hsv':
        result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 轉回 BGR 以便顯示
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    elif mode == 'gray':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == 'lab':
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    else:
        result = img

    # 顯示 HSV 的 H 通道
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    result = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)

    code = '''import cv2

img = cv2.imread('image.jpg')

# BGR 轉灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BGR 轉 HSV (色相、飽和度、明度)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# BGR 轉 LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# BGR 轉 RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR 轉 YCrCb
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# 常見色彩空間：
# - BGR/RGB: 基本色彩
# - HSV: 色相分析、顏色過濾
# - LAB: 與人眼感知相近
# - YCrCb: 影片壓縮'''
    return result, code


def process_alpha_channel(img, params):
    """透明圖層通道"""
    h, w = img.shape[:2]

    # 建立 Alpha 通道 (漸層透明)
    alpha = np.zeros((h, w), dtype=np.uint8)
    for i in range(w):
        alpha[:, i] = int(255 * i / w)

    # 合併為 BGRA
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    # 轉回 BGR 顯示 (用棋盤格表示透明)
    checker = np.zeros((h, w, 3), dtype=np.uint8)
    checker[::20, ::20] = [200, 200, 200]
    checker[10::20, 10::20] = [200, 200, 200]

    alpha_3ch = cv2.merge([alpha, alpha, alpha]) / 255.0
    result = (img * alpha_3ch + checker * (1 - alpha_3ch)).astype(np.uint8)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# BGR 轉 BGRA (加入 Alpha 通道)
bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# 建立 Alpha 遮罩
alpha = np.zeros((h, w), dtype=np.uint8)
cv2.circle(alpha, (w//2, h//2), 100, 255, -1)

# 設定 Alpha 通道
bgra[:, :, 3] = alpha

# 儲存為 PNG (支援透明)
cv2.imwrite('output.png', bgra)

# 讀取有透明通道的圖片
img_with_alpha = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)

# Alpha 混合 (前景與背景合成)
# result = foreground * alpha + background * (1 - alpha)'''
    return result, code


# ===== 幾何轉換 =====

def process_geometric(img, params):
    """幾何轉換"""
    h, w = img.shape[:2]

    # 旋轉
    center = (w // 2, h // 2)
    angle = params.get('angle', 30)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(img, M, (w, h))

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 縮放
scaled = cv2.resize(img, None, fx=0.5, fy=0.5)
# 或指定大小
scaled = cv2.resize(img, (300, 200))

# 旋轉
center = (w // 2, h // 2)
angle = 45  # 逆時針旋轉角度
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, M, (w, h))

# 平移
tx, ty = 100, 50  # x, y 方向位移
M = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, M, (w, h))

# 翻轉
flip_h = cv2.flip(img, 1)   # 水平翻轉
flip_v = cv2.flip(img, 0)   # 垂直翻轉
flip_both = cv2.flip(img, -1)  # 兩者皆翻轉

# 仿射變換
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M, (w, h))

# 透視變換
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M, (300, 300))'''
    return result, code


# ===== 二值化 =====

def process_threshold(img, params):
    """二值化處理"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mode = params.get('mode', 'otsu')

    if mode == 'otsu':
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == 'adaptive_mean':
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    elif mode == 'adaptive_gaussian':
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    else:
        thresh_val = params.get('thresh', 127)
        _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = '''import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 基本二值化
thresh_val = 127
max_val = 255
_, binary = cv2.threshold(img, thresh_val, max_val, cv2.THRESH_BINARY)

# 二值化類型：
# cv2.THRESH_BINARY      - 大於閾值為 max_val，否則為 0
# cv2.THRESH_BINARY_INV  - 反向
# cv2.THRESH_TRUNC       - 大於閾值變成閾值
# cv2.THRESH_TOZERO      - 小於閾值變成 0
# cv2.THRESH_TOZERO_INV  - 大於閾值變成 0

# Otsu 自動閾值
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 自適應閾值 (局部閾值)
adaptive_mean = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,      # 使用鄰域均值
    cv2.THRESH_BINARY,
    11,  # 鄰域大小 (奇數)
    2    # 減去的常數
)

adaptive_gaussian = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 使用高斯加權
    cv2.THRESH_BINARY, 11, 2
)'''
    return result, code


# ===== 平滑處理 =====

def process_blur_mean(img, params):
    """均值濾波"""
    ksize = params.get('ksize', 5)
    result = cv2.blur(img, (ksize, ksize))

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 均值濾波 (Mean Filter / Box Blur)
# 使用鄰域像素的平均值取代中心像素
ksize = 5  # 核心大小
blurred = cv2.blur(img, (ksize, ksize))

# 等同於使用 filter2D 自定義核心
kernel = np.ones((5, 5), np.float32) / 25
blurred = cv2.filter2D(img, -1, kernel)

# 卷積 (Convolution) 概念：
# 將核心 (kernel) 滑動經過影像，
# 每個位置計算核心與影像區域的點積，
# 結果作為輸出影像的像素值。

# 核心大小越大，模糊效果越強'''
    return result, code


def process_blur_box(img, params):
    """方框濾波"""
    ksize = params.get('ksize', 5)
    result = cv2.boxFilter(img, -1, (ksize, ksize), normalize=True)

    code = '''import cv2

img = cv2.imread('image.jpg')

# 方框濾波 (Box Filter)
ksize = 5
# normalize=True: 類似均值濾波 (除以核心面積)
# normalize=False: 不正規化 (可能造成過曝)
blurred = cv2.boxFilter(img, -1, (ksize, ksize), normalize=True)

# ddepth=-1 表示輸出與輸入相同深度
# 也可以指定 cv2.CV_32F 等

# 方框濾波 vs 均值濾波：
# - 方框濾波可以選擇是否正規化
# - 不正規化時可用於計算區域和'''
    return result, code


def process_blur_gaussian(img, params):
    """高斯濾波"""
    ksize = params.get('ksize', 5)
    if ksize % 2 == 0:
        ksize += 1
    result = cv2.GaussianBlur(img, (ksize, ksize), 0)

    code = '''import cv2

img = cv2.imread('image.jpg')

# 高斯濾波 (Gaussian Blur)
# 使用高斯函數作為權重，越靠近中心權重越大
ksize = 5  # 必須是奇數
sigmaX = 0  # 0 表示自動計算
blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX)

# 高斯核心 (5x5 範例):
# [1  4  7  4  1]
# [4 16 26 16  4]
# [7 26 41 26  7] / 273
# [4 16 26 16  4]
# [1  4  7  4  1]

# 高斯濾波優點：
# - 平滑效果自然
# - 保留較多細節
# - 適合去除高斯雜訊'''
    return result, code


def process_blur_median(img, params):
    """中值濾波"""
    ksize = params.get('ksize', 5)
    if ksize % 2 == 0:
        ksize += 1
    result = cv2.medianBlur(img, ksize)

    code = '''import cv2

img = cv2.imread('image.jpg')

# 中值濾波 (Median Blur)
# 使用鄰域像素的中位數取代中心像素
ksize = 5  # 必須是奇數
blurred = cv2.medianBlur(img, ksize)

# 中值濾波的特點：
# - 非線性濾波 (不是加權平均)
# - 對椒鹽雜訊 (Salt and Pepper Noise) 特別有效
# - 能保持邊緣清晰
# - 運算速度較慢

# 適用場景：
# - 去除椒鹽雜訊
# - 保持邊緣的模糊處理'''
    return result, code


def process_blur_bilateral(img, params):
    """雙邊濾波"""
    d = params.get('d', 9)
    result = cv2.bilateralFilter(img, d, 75, 75)

    code = '''import cv2

img = cv2.imread('image.jpg')

# 雙邊濾波 (Bilateral Filter)
# 同時考慮空間距離和顏色相似度
d = 9           # 鄰域直徑
sigmaColor = 75  # 顏色空間標準差
sigmaSpace = 75  # 座標空間標準差
blurred = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

# 雙邊濾波的特點：
# - 邊緣保持濾波 (Edge-Preserving)
# - 只模糊相似顏色的區域
# - 運算速度最慢
# - 可能產生卡通化效果

# 適用場景：
# - 美顏 (磨皮效果)
# - 降噪同時保持邊緣
# - 風格化處理'''
    return result, code


# ===== 形態學 =====

def process_morphology(img, params):
    """形態學操作"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    mode = params.get('mode', 'erosion')

    if mode == 'erosion':
        result = cv2.erode(binary, kernel, iterations=1)
    elif mode == 'dilation':
        result = cv2.dilate(binary, kernel, iterations=1)
    elif mode == 'opening':
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif mode == 'closing':
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif mode == 'gradient':
        result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    else:
        result = binary

    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 建立結構元素 (核心)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 形狀: MORPH_RECT (矩形), MORPH_ELLIPSE (橢圓), MORPH_CROSS (十字)

# 侵蝕 (Erosion) - 縮小白色區域
erosion = cv2.erode(binary, kernel, iterations=1)

# 膨脹 (Dilation) - 擴大白色區域
dilation = cv2.dilate(binary, kernel, iterations=1)

# 開運算 (Opening) = 侵蝕 + 膨脹
# 去除小白點/毛刺
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 閉運算 (Closing) = 膨脹 + 侵蝕
# 填補小黑洞
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 形態學梯度 = 膨脹 - 侵蝕
# 取得邊緣
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

# 頂帽 (Top Hat) = 原圖 - 開運算
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)

# 黑帽 (Black Hat) = 閉運算 - 原圖
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)'''
    return result, code


# ===== 梯度處理 =====

def process_gradient_sobel(img, params):
    """Sobel 梯度"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))

    result = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel 運算子 - 計算梯度
# dx, dy: 對 x, y 方向的導數階數
# ksize: 核心大小 (1, 3, 5, 7)

# X 方向梯度 (垂直邊緣)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# Y 方向梯度 (水平邊緣)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 取絕對值並轉換類型
abs_sobelx = cv2.convertScaleAbs(sobelx)
abs_sobely = cv2.convertScaleAbs(sobely)

# 合併 X 和 Y 方向
sobel = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

# 或計算梯度大小
magnitude = np.sqrt(sobelx**2 + sobely**2)

# Sobel 核心 (3x3):
# X方向:          Y方向:
# [-1  0  1]      [-1 -2 -1]
# [-2  0  2]      [ 0  0  0]
# [-1  0  1]      [ 1  2  1]'''
    return result, code


def process_gradient_scharr(img, params):
    """Scharr 梯度"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

    scharr = np.sqrt(scharrx**2 + scharry**2)
    scharr = np.uint8(np.clip(scharr, 0, 255))

    result = cv2.cvtColor(scharr, cv2.COLOR_GRAY2BGR)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Scharr 運算子 - 比 Sobel 更精確
# 只能計算一階導數

# X 方向梯度
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)

# Y 方向梯度
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)

# 合併
scharr = cv2.addWeighted(
    cv2.convertScaleAbs(scharrx), 0.5,
    cv2.convertScaleAbs(scharry), 0.5, 0
)

# Scharr 核心 (3x3):
# X方向:          Y方向:
# [-3   0   3]    [-3 -10  -3]
# [-10  0  10]    [ 0   0   0]
# [-3   0   3]    [ 3  10   3]

# Scharr vs Sobel:
# - Scharr 對小角度更準確
# - 核心固定為 3x3'''
    return result, code


def process_gradient_laplacian(img, params):
    """Laplacian 梯度"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    result = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Laplacian 運算子 - 二階導數
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

# Laplacian 核心 (3x3):
# [0  1  0]
# [1 -4  1]
# [0  1  0]

# 或 8 鄰域版本:
# [1  1  1]
# [1 -8  1]
# [1  1  1]

# Laplacian 特點:
# - 檢測所有方向的邊緣
# - 對雜訊敏感 (通常先做高斯模糊)
# - 可用於影像銳化

# 銳化公式:
# sharpened = img - laplacian'''
    return result, code


# ===== 邊緣檢測 =====

def process_canny(img, params):
    """Canny 邊緣檢測"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    low = params.get('low', 50)
    high = params.get('high', 150)

    edges = cv2.Canny(gray, low, high)
    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    code = '''import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 建議先做高斯模糊減少雜訊
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Canny 邊緣檢測
threshold1 = 50   # 低閾值
threshold2 = 150  # 高閾值
edges = cv2.Canny(blurred, threshold1, threshold2)

# Canny 演算法步驟：
# 1. 高斯濾波 - 減少雜訊
# 2. 計算梯度 - 使用 Sobel 運算子
# 3. 非極大值抑制 - 保留局部最大值
# 4. 雙閾值檢測 - 分類強邊緣和弱邊緣
# 5. 邊緣連接 - 連接強邊緣相鄰的弱邊緣

# 閾值建議：
# - 高閾值 : 低閾值 = 2:1 或 3:1
# - 可用 Otsu 方法自動計算

# 自動計算閾值
median = np.median(img)
low = int(max(0, 0.7 * median))
high = int(min(255, 1.3 * median))'''
    return result, code


# ===== 影像金字塔 =====

def process_pyramid(img, params):
    """影像金字塔"""
    # 向下取樣
    down1 = cv2.pyrDown(img)
    down2 = cv2.pyrDown(down1)

    # 為了顯示，將它們組合在一起
    h, w = img.shape[:2]
    result = np.zeros((h, w + w//2 + w//4, 3), dtype=np.uint8)
    result[:h, :w] = img
    result[:h//2, w:w+w//2] = down1
    result[:h//4, w+w//2:w+w//2+w//4] = down2

    code = '''import cv2

img = cv2.imread('image.jpg')

# 高斯金字塔 - 向下取樣 (縮小)
# 每次尺寸減半
down1 = cv2.pyrDown(img)      # 1/2
down2 = cv2.pyrDown(down1)    # 1/4
down3 = cv2.pyrDown(down2)    # 1/8

# 高斯金字塔 - 向上取樣 (放大)
# 每次尺寸加倍
up1 = cv2.pyrUp(down1)

# 注意：pyrDown 再 pyrUp 不會完全還原
# 因為縮小時會損失資訊

# 拉普拉斯金字塔 (用於影像融合)
# L = G - pyrUp(pyrDown(G))
laplacian = img - cv2.pyrUp(cv2.pyrDown(img))

# 應用場景：
# - 影像縮放
# - 影像融合 (拉普拉斯金字塔)
# - 多尺度分析
# - 加速影像處理'''
    return result, code


# ===== 輪廓 =====

def process_contours(img, params):
    """影像輪廓"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # 標示最大輪廓的矩形
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 尋找輪廓
# mode: RETR_EXTERNAL (只取外輪廓), RETR_TREE (階層結構)
# method: CHAIN_APPROX_SIMPLE (壓縮), CHAIN_APPROX_NONE (所有點)
contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# 繪製輪廓
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# 輪廓特徵
for cnt in contours:
    # 面積
    area = cv2.contourArea(cnt)

    # 周長
    perimeter = cv2.arcLength(cnt, True)

    # 外接矩形
    x, y, w, h = cv2.boundingRect(cnt)

    # 最小外接矩形 (可旋轉)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    # 最小外接圓
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    # 擬合橢圓
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)

    # 凸包
    hull = cv2.convexHull(cnt)

    # 矩特徵
    moments = cv2.moments(cnt)

    # Hu 矩 (旋轉不變)
    hu = cv2.HuMoments(moments)'''
    return result, code


# ===== 長條圖 =====

def process_histogram(img, params):
    """長條圖處理"""
    # 直方圖均衡化
    if len(img.shape) == 3:
        # 轉換到 YCrCb，只對 Y 通道均衡化
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        result = cv2.equalizeHist(img)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = '''import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 計算直方圖
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 繪製直方圖
plt.plot(hist)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# 直方圖均衡化 (灰階)
equalized = cv2.equalizeHist(gray)

# 彩色影像的直方圖均衡化
# 方法1: 分別處理 BGR 通道 (可能改變顏色)
b, g, r = cv2.split(img)
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)
result = cv2.merge([b_eq, g_eq, r_eq])

# 方法2: 轉換到 YCrCb，只對 Y 通道均衡化 (保持顏色)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# CLAHE (對比度受限自適應直方圖均衡化)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
result = clahe.apply(gray)'''
    return result, code


# ===== 傅立葉轉換 =====

def process_fourier(img, params):
    """傅立葉轉換"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # DFT
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 計算頻譜
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_spectrum = 20 * np.log(magnitude + 1)

    # 正規化到 0-255
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    result = cv2.cvtColor(np.uint8(magnitude_spectrum), cv2.COLOR_GRAY2BGR)

    code = '''import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 離散傅立葉轉換 (DFT)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# 將低頻移到中心
dft_shift = np.fft.fftshift(dft)

# 計算頻譜大小
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum = 20 * np.log(magnitude + 1)

# 低通濾波器 (去除高頻 = 模糊)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), 30, (1, 1), -1)
filtered = dft_shift * mask

# 高通濾波器 (去除低頻 = 邊緣)
mask = np.ones((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), 30, (0, 0), -1)
filtered = dft_shift * mask

# 逆傅立葉轉換
f_ishift = np.fft.ifftshift(filtered)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])'''
    return result, code


# ===== 霍夫轉換 =====

def process_hough_line(img, params):
    """霍夫直線轉換"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    result = img.copy()

    # 機率霍夫變換
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# 標準霍夫變換
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 機率霍夫變換 (更實用)
lines = cv2.HoughLinesP(
    edges,
    rho=1,              # 距離解析度
    theta=np.pi/180,    # 角度解析度
    threshold=50,       # 投票閾值
    minLineLength=50,   # 最小線段長度
    maxLineGap=10       # 最大間隙
)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)'''
    return result, code


def process_hough_circle(img, params):
    """霍夫圓形轉換"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    result = img.copy()

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=30, minRadius=0, maxRadius=0
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 先做模糊減少雜訊
gray = cv2.medianBlur(gray, 5)

# 霍夫圓形轉換
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,  # 檢測方法
    dp=1,                # 累加器解析度比例
    minDist=20,          # 圓心之間最小距離
    param1=50,           # Canny 高閾值
    param2=30,           # 累加器閾值
    minRadius=0,         # 最小半徑
    maxRadius=0          # 最大半徑 (0=無限制)
)

# 繪製圓形
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        # 繪製圓周
        cv2.circle(img, center, radius, (0, 255, 0), 2)
        # 繪製圓心
        cv2.circle(img, center, 2, (0, 0, 255), 3)'''
    return result, code


# ===== 分水嶺 =====

def process_watershed(img, params):
    """分水嶺演算法"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去除雜訊
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 確定背景區域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 確定前景區域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 未知區域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 標記
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 分水嶺
    result = img.copy()
    markers = cv2.watershed(result, markers)
    result[markers == -1] = [0, 0, 255]

    code = '''import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
_, thresh = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 去除雜訊
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 確定背景區域 (膨脹)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 確定前景區域 (距離轉換)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform,
                           0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# 未知區域 (前景和背景之間)
unknown = cv2.subtract(sure_bg, sure_fg)

# 標記連通區域
_, markers = cv2.connectedComponents(sure_fg)

# 背景標記為 1，未知區域標記為 0
markers = markers + 1
markers[unknown == 255] = 0

# 應用分水嶺演算法
markers = cv2.watershed(img, markers)

# 邊界標記為 -1，標示為紅色
img[markers == -1] = [0, 0, 255]'''
    return result, code


# ===== 人臉識別 =====

def process_face_detection(img, params):
    """人臉識別"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 載入人臉分類器
    cascade_file = os.path.join(CASCADE_PATH, 'haarcascade_frontalface_default.xml')

    if not os.path.exists(cascade_file):
        # 如果本地沒有，使用 OpenCV 內建的
        cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(cascade_file)

    # 偵測人臉
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    result = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    code = '''import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 載入預訓練的人臉分類器 (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 也可以載入其他分類器
# eye_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_eye.xml'
# )

# 偵測人臉
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # 每次縮放比例
    minNeighbors=4,    # 最小鄰居數
    minSize=(30, 30)   # 最小人臉大小
)

# 繪製矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 其他可用的 Haar Cascade:
# - haarcascade_frontalface_alt.xml
# - haarcascade_eye.xml
# - haarcascade_smile.xml
# - haarcascade_fullbody.xml
# - haarcascade_upperbody.xml

# 更進階的人臉識別可使用：
# - dlib
# - face_recognition 套件
# - OpenCV DNN (深度學習模型)'''
    return result, code
