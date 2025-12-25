# -*- coding: utf-8 -*-
"""
OpenCV 教學平台 - 影像處理函數模組
包含所有 OpenCV 效果的處理函數、詳細說明和可調參數
"""

import cv2
import numpy as np
import os

# Haar Cascade 路徑
CASCADE_PATH = os.path.join(os.path.dirname(__file__), 'haarcascades')


def get_all_effects():
    """
    取得所有可用的效果列表，包含說明和參數定義

    參數類型說明：
    - slider: 滑桿 (需要 min, max, step)
    - select: 下拉選單 (需要 options)
    - checkbox: 勾選框
    """
    return {
        # ===== 基礎 =====
        'original': {
            'name': '原始影像',
            'category': '基礎',
            'description': '''顯示原始上傳的影像，不做任何處理。

## 基本概念

在 OpenCV 中，影像就是一個 **NumPy 陣列**（ndarray），每個元素代表一個像素值。

### 影像的基本屬性

| 屬性 | 說明 | 範例 |
|------|------|------|
| `shape` | 影像尺寸 | `(480, 640, 3)` = 高480、寬640、3通道 |
| `dtype` | 資料型態 | 通常是 `uint8`（0-255） |
| `size` | 總像素數 | `480 × 640 × 3 = 921600` |

### 讀取影像

```python
import cv2

# 讀取彩色影像
img = cv2.imread('image.jpg')

# 讀取灰階影像
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 顯示影像
cv2.imshow('Window', img)
cv2.waitKey(0)
```

> **注意**：OpenCV 預設使用 **BGR** 順序（藍-綠-紅），而非一般常見的 RGB 順序！

### 延伸學習

- [OpenCV 官方文件 - 讀取影像](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html)
''',
            'params': []
        },
        'channels': {
            'name': '通道拆解 (BGR)',
            'category': '基礎',
            'description': '''將彩色影像拆解為藍(B)、綠(G)、紅(R)三個通道。

## 核心概念

彩色影像是由三個「通道」疊加而成，每個通道記錄該顏色的強度（0-255）。

### 為什麼是 BGR 而非 RGB？

| 順序 | 使用者 | 說明 |
|------|--------|------|
| **BGR** | OpenCV | 歷史原因，早期相機和顯示器使用 BGR |
| **RGB** | Matplotlib、PIL、網頁 | 較直覺的紅-綠-藍順序 |

```python
# BGR 與 RGB 轉換
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
```

### 通道拆解與合併

```python
import cv2

img = cv2.imread('image.jpg')

# 拆解通道
b, g, r = cv2.split(img)

# 合併通道（可調換順序）
merged = cv2.merge([b, g, r])
```

### 實際應用

- **去除紅眼**：降低紅色通道在眼睛區域的值
- **色彩調整**：單獨調整某個通道的對比度
- **特徵提取**：某些物體在特定通道更明顯
- **綠幕去背**：利用綠色通道分離前景

> **提示**：使用 `cv2.split()` 拆解通道會建立資料副本；如果只需讀取，可用 `img[:,:,0]` 切片方式更有效率。
''',
            'params': [
                {'name': 'channel', 'label': '顯示通道', 'type': 'select',
                 'options': [
                     {'value': 'all', 'label': '全部通道'},
                     {'value': 'b', 'label': '藍色通道 (B)'},
                     {'value': 'g', 'label': '綠色通道 (G)'},
                     {'value': 'r', 'label': '紅色通道 (R)'}
                 ], 'default': 'all'}
            ]
        },
        'arithmetic': {
            'name': '影像運算',
            'category': '基礎',
            'description': '''對影像進行加法或減法運算，最基本的像素操作。

## 溢位問題

像素值必須在 0-255 範圍內，超出時會發生溢位：

| 運算方式 | 200 + 100 | 說明 |
|----------|-----------|------|
| `cv2.add()` | **255** | 飽和運算，超過 255 就設為 255 |
| `img + 100` | **44** | 溢位！(300 % 256 = 44) |

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')

# ✅ 正確做法：使用 cv2.add()
bright = cv2.add(img, 50)  # 整體加亮

# ❌ 危險做法：直接加法可能溢位
# bright = img + 50  # 可能產生奇怪的結果
```

### 亮度與對比度調整

更完整的亮度/對比度調整公式：

```python
# new_pixel = α × old_pixel + β
# α：對比度 (1.0 = 不變, >1 增加對比)
# β：亮度 (+50 變亮, -50 變暗)

result = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
```

### 實際應用

- **曝光校正**：照片太暗時增加亮度
- **HDR 合成**：多張不同曝光照片合成
- **影像差異**：兩張圖相減找出變化區域

> **提示**：`cv2.subtract()` 會將負數設為 0，同樣避免溢位問題。
''',
            'params': [
                {'name': 'operation', 'label': '運算類型', 'type': 'select',
                 'options': [
                     {'value': 'add', 'label': '加法 (變亮)'},
                     {'value': 'subtract', 'label': '減法 (變暗)'}
                 ], 'default': 'add'},
                {'name': 'value', 'label': '調整值', 'type': 'slider',
                 'min': 0, 'max': 100, 'step': 5, 'default': 50}
            ]
        },
        'weighted': {
            'name': '影像加權和',
            'category': '基礎',
            'description': '''將兩張影像按權重比例混合，實現影像融合效果。

## 公式說明

```
dst = src1 × α + src2 × β + γ
```

| 參數 | 意義 | 範圍 |
|------|------|------|
| **α** (alpha) | 第一張圖的權重 | 0.0 ~ 1.0 |
| **β** (beta) | 第二張圖的權重 | 通常 = 1-α |
| **γ** (gamma) | 亮度調整值 | -255 ~ 255 |

### 基本用法

```python
import cv2

img1 = cv2.imread('photo.jpg')
img2 = cv2.imread('watermark.png')

# α=0.7 表示 70% 原圖 + 30% 浮水印
result = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
```

### 淡入淡出效果

```python
# 影片轉場效果：從 img1 漸變到 img2
for alpha in np.linspace(1, 0, 30):  # 30 個步驟
    beta = 1 - alpha
    frame = cv2.addWeighted(img1, alpha, img2, beta, 0)
    cv2.imshow('Transition', frame)
    cv2.waitKey(50)
```

### 實際應用

- **半透明浮水印**：logo 疊加在影片上
- **全景圖拼接**：重疊區域的平滑過渡
- **影片淡入淡出**：轉場特效
- **影像比對**：快速切換兩張圖比較差異

> **注意**：兩張圖的尺寸必須相同！若不同需先用 `cv2.resize()` 調整。
''',
            'params': [
                {'name': 'alpha', 'label': '原圖權重 (α)', 'type': 'slider',
                 'min': 0, 'max': 1, 'step': 0.05, 'default': 0.7},
                {'name': 'blend_type', 'label': '混合圖類型', 'type': 'select',
                 'options': [
                     {'value': 'gradient_h', 'label': '水平漸層 (左→右)'},
                     {'value': 'gradient_v', 'label': '垂直漸層 (上→下)'},
                     {'value': 'gradient_d', 'label': '對角漸層'},
                     {'value': 'radial', 'label': '放射漸層 (中心→邊緣)'},
                     {'value': 'solid', 'label': '純色背景'},
                     {'value': 'checker', 'label': '棋盤格'},
                     {'value': 'noise', 'label': '隨機雜訊'}
                 ], 'default': 'gradient_h'},
                {'name': 'color_r', 'label': '顏色 R', 'type': 'slider',
                 'min': 0, 'max': 255, 'step': 5, 'default': 255},
                {'name': 'color_g', 'label': '顏色 G', 'type': 'slider',
                 'min': 0, 'max': 255, 'step': 5, 'default': 255},
                {'name': 'color_b', 'label': '顏色 B', 'type': 'slider',
                 'min': 0, 'max': 255, 'step': 5, 'default': 255}
            ]
        },
        'bitwise': {
            'name': '逐位元邏輯運算',
            'category': '基礎',
            'description': '''對每個像素的二進位值進行邏輯運算，是影像合成的核心技術。

## 四種運算

| 運算 | 符號 | 規則 | 主要用途 |
|------|------|------|----------|
| **AND** | `&` | 兩者都為 1 才是 1 | 遮罩提取 |
| **OR** | `｜` | 任一為 1 就是 1 | 合併亮部 |
| **XOR** | `^` | 兩者不同才是 1 | 找差異、加密 |
| **NOT** | `~` | 0 變 1、1 變 0 | 負片效果 |

### 遮罩提取範例

```python
import cv2
import numpy as np

img = cv2.imread('photo.jpg')
h, w = img.shape[:2]

# 建立圓形遮罩（白色圓、黑色背景）
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (w//2, h//2), 100, 255, -1)

# AND 運算：只保留圓形區域
result = cv2.bitwise_and(img, img, mask=mask)
```

### Logo 合成（去背疊加）

```python
# 經典的 Logo 疊加技巧
logo = cv2.imread('logo.png')
roi = img[0:logo.shape[0], 0:logo.shape[1]]

# 1. 建立 Logo 遮罩
gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# 2. 在 ROI 中挖洞
bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# 3. 取出 Logo 前景
fg = cv2.bitwise_and(logo, logo, mask=mask)

# 4. 合併
dst = cv2.add(bg, fg)
img[0:logo.shape[0], 0:logo.shape[1]] = dst
```

### 位元運算示意

```
像素值 200 = 11001000
像素值  50 = 00110010
─────────────────────
AND 結果   = 00000000 (0)
OR 結果    = 11111010 (250)
XOR 結果   = 11111010 (250)
```

> **提示**：遮罩（mask）通常是灰階圖，白色區域（255）會被保留，黑色區域（0）會被遮蔽。
''',
            'params': [
                {'name': 'operation', 'label': '運算類型', 'type': 'select',
                 'options': [
                     {'value': 'and', 'label': 'AND (交集/遮罩提取)'},
                     {'value': 'or', 'label': 'OR (聯集/疊加亮部)'},
                     {'value': 'xor', 'label': 'XOR (互斥/找差異)'},
                     {'value': 'not', 'label': 'NOT (反轉/負片)'}
                 ], 'default': 'and'},
                {'name': 'shape', 'label': '遮罩形狀', 'type': 'select',
                 'options': [
                     {'value': 'circle', 'label': '圓形'},
                     {'value': 'rectangle', 'label': '矩形'},
                     {'value': 'ellipse', 'label': '橢圓形'},
                     {'value': 'triangle', 'label': '三角形'}
                 ], 'default': 'circle'},
                {'name': 'mask_size', 'label': '遮罩大小 (%)', 'type': 'slider',
                 'min': 10, 'max': 90, 'step': 5, 'default': 50},
                {'name': 'center_x', 'label': '中心X位置 (%)', 'type': 'slider',
                 'min': 10, 'max': 90, 'step': 5, 'default': 50},
                {'name': 'center_y', 'label': '中心Y位置 (%)', 'type': 'slider',
                 'min': 10, 'max': 90, 'step': 5, 'default': 50},
                {'name': 'show_mask', 'label': '顯示遮罩', 'type': 'checkbox', 'default': False}
            ]
        },
        'bit_plane': {
            'name': '位元平面分解',
            'category': '基礎',
            'description': '''將每個像素拆解成 8 個位元平面，揭示影像的二進位結構。

## 核心概念

每個灰階像素值（0-255）用 8 個位元表示：

```
像素值 200 = 1 1 0 0 1 0 0 0
             ↑             ↑
           平面7         平面0
           (MSB)         (LSB)
           權重128       權重1
```

### 各平面的意義

| 平面 | 權重 | 資訊含量 | 視覺效果 |
|------|------|----------|----------|
| 7 (MSB) | 128 | **最高** | 清晰的影像輪廓 |
| 6 | 64 | 高 | 仍可辨識結構 |
| 5 | 32 | 中 | 輪廓開始模糊 |
| 4 | 16 | 中 | 細節消失 |
| 3 | 8 | 低 | 雜訊增加 |
| 2 | 4 | 低 | 幾乎是雜訊 |
| 1 | 2 | 很低 | 隨機圖案 |
| 0 (LSB) | 1 | **最低** | 純雜訊 |

### 提取位元平面

```python
import cv2
import numpy as np

gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 提取第 7 個位元平面（MSB）
plane_7 = (gray >> 7) & 1
plane_7 = plane_7 * 255  # 轉成可視化

# 提取第 0 個位元平面（LSB）
plane_0 = gray & 1
plane_0 = plane_0 * 255
```

### 實際應用

- **影像隱寫術**：在 LSB 藏入秘密訊息，肉眼無法察覺
- **影像壓縮**：只保留高位元平面
- **數位浮水印**：在特定平面嵌入版權標記
- **影像分析**：檢測竄改或合成痕跡

> **有趣事實**：高位元平面（7-4）就包含了影像 94% 以上的資訊！
''',
            'params': [
                {'name': 'plane', 'label': '位元平面 (0=LSB, 7=MSB)', 'type': 'slider',
                 'min': 0, 'max': 7, 'step': 1, 'default': 7},
                {'name': 'show_all', 'label': '顯示所有平面', 'type': 'checkbox', 'default': False}
            ]
        },
        'encrypt': {
            'name': '影像加密與解密',
            'category': '基礎',
            'description': '''利用 XOR 運算的對稱特性進行影像加密與解密。

## XOR 加密原理

XOR 有個神奇的數學特性：**做兩次會還原**

```
A ⊕ B ⊕ B = A
```

| 步驟 | 運算 | 結果 |
|------|------|------|
| 原始像素 | 200 | `11001000` |
| 加密 (⊕ 金鑰 50) | 200 ⊕ 50 | **250** `11111010` |
| 解密 (⊕ 金鑰 50) | 250 ⊕ 50 | **200** `11001000` ✓ |

### 實作範例

```python
import cv2
import numpy as np

img = cv2.imread('secret.jpg')

# 產生金鑰（用固定種子確保可重現）
np.random.seed(42)
key = np.random.randint(0, 256, img.shape, dtype=np.uint8)

# 加密
encrypted = cv2.bitwise_xor(img, key)
cv2.imwrite('encrypted.png', encrypted)

# 解密（用同樣的種子產生同樣的金鑰）
np.random.seed(42)
key = np.random.randint(0, 256, img.shape, dtype=np.uint8)
decrypted = cv2.bitwise_xor(encrypted, key)
```

### 金鑰種子的重要性

- **種子相同** → 產生的隨機序列相同 → 金鑰相同
- 種子就是「密碼」，只有知道種子才能解密
- 種子不同 → 解密出來是亂碼

### 安全性說明

⚠️ **這是教學示範！** 真正的影像加密應使用：
- AES-256 等專業加密演算法
- 安全的金鑰交換機制
- 加密函式庫如 `cryptography` 或 `PyCryptodome`

> **有趣事實**：這種 XOR 加密在 1970 年代曾被用於真正的加密系統，稱為「一次性密碼本」(One-Time Pad)，理論上是無法破解的！
''',
            'params': [
                {'name': 'seed', 'label': '金鑰種子', 'type': 'slider',
                 'min': 1, 'max': 100, 'step': 1, 'default': 42},
                {'name': 'display', 'label': '顯示內容', 'type': 'select',
                 'options': [
                     {'value': 'encrypted', 'label': '加密後的圖'},
                     {'value': 'decrypted', 'label': '解密還原的圖'},
                     {'value': 'both', 'label': '加密與解密並排'}
                 ], 'default': 'both'}
            ]
        },

        # ===== 色彩 =====
        'color_space': {
            'name': '色彩空間轉換',
            'category': '色彩',
            'description': '''不同的色彩空間用不同方式描述顏色，各有最適合的應用場景。

## 色彩空間比較

| 色彩空間 | 組成 | 最佳用途 |
|----------|------|----------|
| **BGR/RGB** | 藍/綠/紅 | 顯示、儲存 |
| **HSV** | 色相/飽和度/明度 | 顏色過濾、物體追蹤 |
| **HLS** | 色相/亮度/飽和度 | 類似 HSV |
| **LAB** | 亮度/綠紅軸/藍黃軸 | 色彩校正、色差計算 |
| **YCrCb** | 亮度/紅色度/藍色度 | 膚色偵測、影片壓縮 |
| **Gray** | 單通道灰階 | 邊緣檢測、二值化 |

### HSV - 顏色過濾的首選

```python
import cv2
import numpy as np

img = cv2.imread('fruits.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 過濾紅色（注意：紅色在 H 軸兩端 0-10 和 170-180）
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

# 只保留紅色區域
result = cv2.bitwise_and(img, img, mask=mask)
```

### 常見顏色的 H 值範圍

| 顏色 | H 值範圍 | 說明 |
|------|----------|------|
| 紅色 | 0-10, 170-180 | 跨越兩端 |
| 橙色 | 10-25 | |
| 黃色 | 25-35 | |
| 綠色 | 35-85 | |
| 藍色 | 85-130 | |
| 紫色 | 130-170 | |

### LAB - 接近人眼感知

```python
# 計算兩個顏色的視覺差異
lab1 = cv2.cvtColor(color1, cv2.COLOR_BGR2LAB)
lab2 = cv2.cvtColor(color2, cv2.COLOR_BGR2LAB)

# 歐氏距離 = 視覺差異
delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2))
```

### YCrCb - 膚色偵測

```python
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# 膚色範圍（經驗值）
lower = np.array([0, 133, 77])
upper = np.array([255, 173, 127])
mask = cv2.inRange(ycrcb, lower, upper)
```

> **提示**：做顏色過濾時，建議先轉 HSV，因為 H 通道直接對應顏色種類，不受光線明暗影響！
''',
            'params': [
                {'name': 'space', 'label': '色彩空間', 'type': 'select',
                 'options': [
                     {'value': 'gray', 'label': '灰階 (Grayscale)'},
                     {'value': 'rgb', 'label': 'RGB（調換 B 和 R）'},
                     {'value': 'hsv', 'label': 'HSV 完整顯示'},
                     {'value': 'hsv_h', 'label': 'HSV - H 色相（色彩地圖）'},
                     {'value': 'hsv_s', 'label': 'HSV - S 飽和度'},
                     {'value': 'hsv_v', 'label': 'HSV - V 明度'},
                     {'value': 'hls', 'label': 'HLS 完整顯示'},
                     {'value': 'hls_h', 'label': 'HLS - H 色相'},
                     {'value': 'hls_l', 'label': 'HLS - L 亮度'},
                     {'value': 'hls_s', 'label': 'HLS - S 飽和度'},
                     {'value': 'lab', 'label': 'LAB 完整顯示'},
                     {'value': 'lab_l', 'label': 'LAB - L 亮度'},
                     {'value': 'lab_a', 'label': 'LAB - A 綠紅軸'},
                     {'value': 'lab_b', 'label': 'LAB - B 藍黃軸'},
                     {'value': 'ycrcb', 'label': 'YCrCb 完整顯示'},
                     {'value': 'ycrcb_y', 'label': 'YCrCb - Y 亮度'},
                     {'value': 'ycrcb_cr', 'label': 'YCrCb - Cr 紅色色度'},
                     {'value': 'ycrcb_cb', 'label': 'YCrCb - Cb 藍色色度'}
                 ], 'default': 'hsv_h'},
                {'name': 'colorize_h', 'label': 'H通道用彩色顯示', 'type': 'checkbox', 'default': True}
            ]
        },

        # ===== 幾何 =====
        'geometric': {
            'name': '幾何轉換',
            'category': '幾何',
            'description': '''對影像進行縮放、旋轉、翻轉等空間變換。

## 基本操作

### 縮放 (Resize)

```python
import cv2

img = cv2.imread('image.jpg')

# 指定目標尺寸
resized = cv2.resize(img, (640, 480))

# 指定縮放比例
half = cv2.resize(img, None, fx=0.5, fy=0.5)
double = cv2.resize(img, None, fx=2, fy=2)
```

### 插值方法比較

| 方法 | 速度 | 品質 | 適用場景 |
|------|------|------|----------|
| `INTER_NEAREST` | 最快 | 低 | 放大像素藝術 |
| `INTER_LINEAR` | 快 | 中 | **預設**、一般縮放 |
| `INTER_CUBIC` | 中 | 高 | 放大照片 |
| `INTER_AREA` | 中 | 高 | **縮小**時最佳 |

### 旋轉 (Rotate)

```python
h, w = img.shape[:2]
center = (w // 2, h // 2)

# 建立旋轉矩陣（中心點、角度、縮放）
M = cv2.getRotationMatrix2D(center, 45, 1.0)

# 套用旋轉
rotated = cv2.warpAffine(img, M, (w, h))
```

### 翻轉 (Flip)

```python
# flipCode: 0=垂直翻轉, 1=水平翻轉, -1=雙向
horizontal = cv2.flip(img, 1)
vertical = cv2.flip(img, 0)
both = cv2.flip(img, -1)
```

### 實際應用

- **資料增強**：訓練 AI 時隨機旋轉、翻轉增加樣本
- **影像校正**：修正歪斜的文件照片
- **縮圖產生**：為網頁產生不同尺寸的圖片

> **提示**：旋轉後圖片會有黑邊，可設定 `borderValue` 或裁切處理。
''',
            'params': [
                {'name': 'transform', 'label': '轉換類型', 'type': 'select',
                 'options': [
                     {'value': 'rotate', 'label': '旋轉'},
                     {'value': 'scale', 'label': '縮放'},
                     {'value': 'flip_h', 'label': '水平翻轉'},
                     {'value': 'flip_v', 'label': '垂直翻轉'},
                     {'value': 'flip_both', 'label': '雙向翻轉'}
                 ], 'default': 'rotate'},
                {'name': 'angle', 'label': '旋轉角度', 'type': 'slider',
                 'min': -180, 'max': 180, 'step': 15, 'default': 45},
                {'name': 'scale', 'label': '縮放比例', 'type': 'slider',
                 'min': 0.1, 'max': 2.0, 'step': 0.1, 'default': 1.0}
            ]
        },

        # ===== 二值化 =====
        'threshold': {
            'name': '二值化處理',
            'category': '二值化',
            'description': '''將灰階影像轉為純黑白（0 或 255），是許多影像處理的前置步驟。

## 三種二值化方法

### 1. 固定閾值

最簡單的方法，所有像素與固定值比較：

```python
import cv2

gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 像素值 > 127 設為 255，否則設為 0
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 反向：像素值 > 127 設為 0，否則設為 255
_, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
```

### 2. Otsu 自動閾值

自動找出最佳分界點，適合雙峰直方圖：

```python
# thresh 參數設 0，讓 Otsu 自動計算
ret, binary = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f'Otsu 計算的閾值: {ret}')
```

### 3. 自適應閾值

根據局部區域計算閾值，適合光線不均的影像：

```python
# adaptiveMethod: MEAN（區域平均）或 GAUSSIAN（加權平均）
# blockSize: 鄰域大小（奇數）
# C: 從平均值減去的常數

adaptive = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,  # blockSize
    2    # C
)
```

### 方法比較

| 方法 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| 固定閾值 | 簡單快速 | 需手動調整 | 光線均勻 |
| Otsu | 自動計算 | 需雙峰分布 | 前景背景明顯 |
| 自適應 | 處理光線不均 | 較慢 | 文件掃描、陰影 |

> **提示**：處理文件掃描時，先做高斯模糊去噪，再用自適應閾值效果最好！
''',
            'params': [
                {'name': 'method', 'label': '二值化方法', 'type': 'select',
                 'options': [
                     {'value': 'binary', 'label': '固定閾值'},
                     {'value': 'binary_inv', 'label': '固定閾值 (反向)'},
                     {'value': 'otsu', 'label': 'Otsu 自動閾值'},
                     {'value': 'adaptive_mean', 'label': '自適應 (均值)'},
                     {'value': 'adaptive_gaussian', 'label': '自適應 (高斯)'}
                 ], 'default': 'otsu'},
                {'name': 'thresh', 'label': '閾值 (固定閾值用)', 'type': 'slider',
                 'min': 0, 'max': 255, 'step': 5, 'default': 127},
                {'name': 'block_size', 'label': '區塊大小 (自適應用)', 'type': 'slider',
                 'min': 3, 'max': 51, 'step': 2, 'default': 11}
            ]
        },

        # ===== 平滑處理 =====
        'blur_mean': {
            'name': '均值濾波',
            'category': '平滑處理',
            'description': '''最基本的模糊方法，用鄰域像素的平均值取代中心像素。

## 卷積原理

均值濾波是一種「卷積」操作，使用一個核心（kernel）滑過影像：

```
3×3 均值核心:
┌─────┬─────┬─────┐
│ 1/9 │ 1/9 │ 1/9 │
├─────┼─────┼─────┤
│ 1/9 │ 1/9 │ 1/9 │
├─────┼─────┼─────┤
│ 1/9 │ 1/9 │ 1/9 │
└─────┴─────┴─────┘
```

新像素值 = 周圍 9 個像素的平均

### 程式碼

```python
import cv2

img = cv2.imread('image.jpg')

# 使用 5×5 核心
blurred = cv2.blur(img, (5, 5))

# 等同於
# kernel = np.ones((5, 5), np.float32) / 25
# blurred = cv2.filter2D(img, -1, kernel)
```

### 核心大小的影響

| 核心大小 | 效果 | 用途 |
|----------|------|------|
| 3×3 | 輕微模糊 | 輕微去噪 |
| 5×5 | 中等模糊 | 一般去噪 |
| 11×11+ | 強烈模糊 | 背景虛化 |

### 優缺點

- **優點**：計算簡單、速度快
- **缺點**：模糊邊緣、無法保持細節

> **提示**：核心大小必須是奇數（3, 5, 7...），確保有中心點！
''',
            'params': [
                {'name': 'ksize', 'label': '核心大小', 'type': 'slider',
                 'min': 3, 'max': 31, 'step': 2, 'default': 5}
            ]
        },
        'blur_gaussian': {
            'name': '高斯濾波',
            'category': '平滑處理',
            'description': '''使用高斯分布作為權重的模糊，比均值濾波更自然。

## 高斯核心

權重呈鐘型分布，越靠近中心權重越大：

```
3×3 高斯核心（示意）:
┌─────┬─────┬─────┐
│  1  │  2  │  1  │
├─────┼─────┼─────┤
│  2  │  4  │  2  │  ÷ 16
├─────┼─────┼─────┤
│  1  │  2  │  1  │
└─────┴─────┴─────┘
```

### 程式碼

```python
import cv2

img = cv2.imread('image.jpg')

# ksize: 核心大小（奇數）
# sigmaX: 標準差（0 = 自動根據 ksize 計算）
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 可分別設定 X 和 Y 方向的 sigma
blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5, sigmaY=1.5)
```

### Sigma 的影響

| Sigma | 效果 | 說明 |
|-------|------|------|
| 小 (< 1) | 輕微模糊 | 權重集中在中心 |
| 中 (1-2) | 適中模糊 | 常用範圍 |
| 大 (> 2) | 強烈模糊 | 權重分散 |

### 與均值濾波比較

- **均值濾波**：所有像素權重相同，會產生「方塊感」
- **高斯濾波**：中心權重大，過渡更平滑自然

### 實際應用

- 去除高斯雜訊（相機感光元件產生）
- 邊緣檢測前的預處理
- 建立「LoG」(Laplacian of Gaussian) 算子

> **提示**：高斯濾波是影像處理最常用的預處理步驟！
''',
            'params': [
                {'name': 'ksize', 'label': '核心大小', 'type': 'slider',
                 'min': 3, 'max': 31, 'step': 2, 'default': 5},
                {'name': 'sigma', 'label': 'Sigma (0=自動)', 'type': 'slider',
                 'min': 0, 'max': 10, 'step': 0.5, 'default': 0}
            ]
        },
        'blur_median': {
            'name': '中值濾波',
            'category': '平滑處理',
            'description': '''用鄰域像素的中位數取代中心像素，對椒鹽雜訊特別有效。

## 中位數原理

將鄰域像素排序，取中間值：

```
鄰域像素: [10, 12, 15, 200, 18, 14, 11, 13, 16]
         （200 是椒鹽雜訊）

排序後:   [10, 11, 12, 13, 14, 15, 16, 18, 200]
                         ↑
                      中位數 = 14
```

雜訊值 200 被「忽略」了！

### 程式碼

```python
import cv2

img = cv2.imread('noisy_image.jpg')

# ksize 必須是奇數
median = cv2.medianBlur(img, 5)
```

### 與均值濾波比較

| 特性 | 均值濾波 | 中值濾波 |
|------|----------|----------|
| 雜訊處理 | 雜訊被「平均」進去 | 雜訊被「忽略」 |
| 邊緣保持 | 模糊邊緣 | 較好保持邊緣 |
| 計算速度 | 快 | 較慢（需排序） |
| 最佳用途 | 高斯雜訊 | **椒鹽雜訊** |

### 椒鹽雜訊

隨機出現的純黑（0）或純白（255）像素：
- 可能來自感測器故障
- 傳輸錯誤
- 損壞的儲存媒體

```python
# 模擬椒鹽雜訊
import numpy as np

noisy = img.copy()
prob = 0.02  # 2% 的像素有雜訊

# 加入「鹽」（白點）
salt = np.random.random(img.shape[:2]) < prob/2
noisy[salt] = 255

# 加入「胡椒」（黑點）
pepper = np.random.random(img.shape[:2]) < prob/2
noisy[pepper] = 0
```

> **提示**：中值濾波是去除椒鹽雜訊的首選方法！
''',
            'params': [
                {'name': 'ksize', 'label': '核心大小', 'type': 'slider',
                 'min': 3, 'max': 31, 'step': 2, 'default': 5}
            ]
        },
        'blur_bilateral': {
            'name': '雙邊濾波',
            'category': '平滑處理',
            'description': '''同時考慮空間距離和顏色相似度的濾波，能保持邊緣同時平滑區域。

## 工作原理

雙邊濾波結合兩種權重：

1. **空間權重**：像素越近，權重越大（類似高斯）
2. **顏色權重**：顏色越接近，權重越大

```
邊緣處：
┌───┬───┬───┐
│150│155│152│  ← 顏色相近，會被平均
├───┼───┼───┤
│148│▓▓▓│ 50│  ← 顏色差異大，不會被平均
├───┼───┼───┤
│153│ 48│ 52│
└───┴───┴───┘
```

### 程式碼

```python
import cv2

img = cv2.imread('portrait.jpg')

# d: 鄰域直徑（-1 = 自動根據 sigmaSpace）
# sigmaColor: 顏色空間的 sigma
# sigmaSpace: 座標空間的 sigma
bilateral = cv2.bilateralFilter(img, d=9,
                                sigmaColor=75,
                                sigmaSpace=75)
```

### 參數說明

| 參數 | 建議值 | 說明 |
|------|--------|------|
| d | 5-9 | 過大會很慢 |
| sigmaColor | 50-100 | 越大，顏色差異容忍度越高 |
| sigmaSpace | 50-100 | 越大，遠處像素影響越大 |

### 實際應用

- **美顏磨皮**：平滑皮膚但保持五官輪廓
- **卡通化**：配合邊緣檢測製作卡通效果
- **HDR 處理**：保持細節的色調映射

### 與其他濾波比較

| 濾波方法 | 邊緣保持 | 速度 |
|----------|----------|------|
| 均值 | ❌ 差 | 最快 |
| 高斯 | ❌ 差 | 快 |
| 中值 | ⚠️ 中 | 中 |
| 雙邊 | ✅ 好 | **慢** |

> **提示**：雙邊濾波計算量大，不適合即時影片處理！
''',
            'params': [
                {'name': 'd', 'label': '鄰域直徑', 'type': 'slider',
                 'min': 3, 'max': 15, 'step': 2, 'default': 9},
                {'name': 'sigma_color', 'label': '顏色 Sigma', 'type': 'slider',
                 'min': 10, 'max': 150, 'step': 10, 'default': 75},
                {'name': 'sigma_space', 'label': '空間 Sigma', 'type': 'slider',
                 'min': 10, 'max': 150, 'step': 10, 'default': 75}
            ]
        },

        # ===== 形態學 =====
        'morphology': {
            'name': '形態學操作',
            'category': '形態學',
            'description': '''對二值化影像的形狀進行變換，常用於去噪、分割、特徵提取。

## 基本操作

### 侵蝕 (Erosion)
縮小白色區域，去除小白點

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
erosion = cv2.erode(binary, kernel, iterations=1)
```

### 膨脹 (Dilation)
擴大白色區域，填補小黑洞

```python
dilation = cv2.dilate(binary, kernel, iterations=1)
```

## 複合操作

| 操作 | 公式 | 效果 |
|------|------|------|
| **開運算** | 先侵蝕後膨脹 | 去除小白點、平滑邊緣 |
| **閉運算** | 先膨脹後侵蝕 | 填補小黑洞、連接斷裂 |
| **梯度** | 膨脹 - 侵蝕 | 取得物體邊緣 |
| **頂帽** | 原圖 - 開運算 | 提取亮小物體 |
| **黑帽** | 閉運算 - 原圖 | 提取暗小物體 |

```python
# 開運算：去除雜點
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 閉運算：填補空洞
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 形態學梯度：邊緣
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
```

## 核心形狀

```python
# 矩形核心
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 橢圓核心（更平滑）
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 十字核心
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
```

### 實際應用

- **文字識別前處理**：去除雜點、連接斷裂筆畫
- **細胞計數**：分離相連細胞
- **車牌定位**：連接車牌上的字元區域
- **指紋增強**：連接斷裂的指紋線

> **提示**：形態學操作通常在二值化影像上進行！
''',
            'params': [
                {'name': 'operation', 'label': '操作類型', 'type': 'select',
                 'options': [
                     {'value': 'erosion', 'label': '侵蝕 (Erosion)'},
                     {'value': 'dilation', 'label': '膨脹 (Dilation)'},
                     {'value': 'opening', 'label': '開運算 (Opening)'},
                     {'value': 'closing', 'label': '閉運算 (Closing)'},
                     {'value': 'gradient', 'label': '形態學梯度'},
                     {'value': 'tophat', 'label': '頂帽 (Top Hat)'},
                     {'value': 'blackhat', 'label': '黑帽 (Black Hat)'}
                 ], 'default': 'erosion'},
                {'name': 'kernel_size', 'label': '核心大小', 'type': 'slider',
                 'min': 3, 'max': 21, 'step': 2, 'default': 5},
                {'name': 'kernel_shape', 'label': '核心形狀', 'type': 'select',
                 'options': [
                     {'value': 'rect', 'label': '矩形'},
                     {'value': 'ellipse', 'label': '橢圓形'},
                     {'value': 'cross', 'label': '十字形'}
                 ], 'default': 'rect'},
                {'name': 'iterations', 'label': '迭代次數', 'type': 'slider',
                 'min': 1, 'max': 10, 'step': 1, 'default': 1}
            ]
        },

        # ===== 梯度 =====
        'gradient_sobel': {
            'name': 'Sobel 梯度',
            'category': '梯度',
            'description': '''計算影像的一階導數（梯度），用於邊緣檢測和特徵提取。

## Sobel 運算子

使用 3×3 的卷積核心計算梯度：

```
X 方向（偵測垂直邊緣）：    Y 方向（偵測水平邊緣）：
┌────┬────┬────┐          ┌────┬────┬────┐
│ -1 │  0 │ +1 │          │ -1 │ -2 │ -1 │
├────┼────┼────┤          ├────┼────┼────┤
│ -2 │  0 │ +2 │          │  0 │  0 │  0 │
├────┼────┼────┤          ├────┼────┼────┤
│ -1 │  0 │ +1 │          │ +1 │ +2 │ +1 │
└────┴────┴────┘          └────┴────┴────┘
```

### 程式碼

```python
import cv2
import numpy as np

gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 計算 X 方向梯度（垂直邊緣）
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# 計算 Y 方向梯度（水平邊緣）
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 計算梯度大小
magnitude = np.sqrt(sobelx**2 + sobely**2)

# 轉換成 uint8
result = cv2.convertScaleAbs(magnitude)
```

### 梯度方向

```python
# 計算梯度方向（弧度）
angle = np.arctan2(sobely, sobelx)
```

### 核心大小的影響

| ksize | 效果 | 說明 |
|-------|------|------|
| 1 | 最銳利 | 3×1 或 1×3 核心 |
| 3 | 標準 | 最常用 |
| 5, 7 | 較平滑 | 對雜訊較不敏感 |

### 與 Canny 的關係

Canny 邊緣檢測內部使用 Sobel 計算梯度：

```
Canny = 高斯模糊 + Sobel梯度 + 非極大值抑制 + 雙閾值
```

### 實際應用

- 邊緣方向分析
- 影像銳化（Unsharp Masking）
- 光流計算的基礎
- 作為 Canny 的預備步驟

> **提示**：Sobel 計算結果可能超過 0-255，記得用 `cv2.CV_64F` 並轉換！
''',
            'params': [
                {'name': 'direction', 'label': '梯度方向', 'type': 'select',
                 'options': [
                     {'value': 'both', 'label': 'X + Y 合併'},
                     {'value': 'x', 'label': 'X 方向 (垂直邊緣)'},
                     {'value': 'y', 'label': 'Y 方向 (水平邊緣)'}
                 ], 'default': 'both'},
                {'name': 'ksize', 'label': '核心大小', 'type': 'select',
                 'options': [
                     {'value': '1', 'label': '1'},
                     {'value': '3', 'label': '3'},
                     {'value': '5', 'label': '5'},
                     {'value': '7', 'label': '7'}
                 ], 'default': '3'}
            ]
        },
        'gradient_laplacian': {
            'name': 'Laplacian 梯度',
            'category': '梯度',
            'description': 'Laplacian 運算子計算二階導數，可同時檢測所有方向的邊緣。對雜訊敏感，通常先做高斯模糊。可用於影像銳化。',
            'params': [
                {'name': 'ksize', 'label': '核心大小', 'type': 'select',
                 'options': [
                     {'value': '1', 'label': '1'},
                     {'value': '3', 'label': '3'},
                     {'value': '5', 'label': '5'}
                 ], 'default': '3'}
            ]
        },

        # ===== 邊緣檢測 =====
        'canny': {
            'name': 'Canny 邊緣檢測',
            'category': '邊緣檢測',
            'description': '''業界最常用的邊緣檢測演算法，由 John Canny 於 1986 年提出。

## 演算法步驟

```
原始影像 → 高斯濾波 → 計算梯度 → 非極大值抑制 → 雙閾值檢測 → 邊緣
```

### 1. 高斯濾波
去除雜訊，避免誤判

### 2. 計算梯度
使用 Sobel 運算子計算 X、Y 方向梯度

### 3. 非極大值抑制
只保留梯度方向上的局部最大值，讓邊緣變細

### 4. 雙閾值檢測
- **強邊緣**：梯度 > 高閾值 → 確定是邊緣
- **弱邊緣**：低閾值 < 梯度 < 高閾值 → 待定
- **非邊緣**：梯度 < 低閾值 → 捨棄

### 5. 邊緣連接
弱邊緣如果與強邊緣相連，就保留；否則捨棄

### 程式碼

```python
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 先做高斯模糊（建議）
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny 邊緣檢測
# threshold1: 低閾值
# threshold2: 高閾值
edges = cv2.Canny(blur, 50, 150)
```

### 閾值設定建議

| 比例 | 適用場景 |
|------|----------|
| 2:1 (如 50:100) | 一般場景 |
| 3:1 (如 50:150) | 較多細節 |

### 自動閾值

```python
# 使用中位數計算閾值
median = np.median(gray)
lower = int(max(0, 0.7 * median))
upper = int(min(255, 1.3 * median))
edges = cv2.Canny(blur, lower, upper)
```

### 實際應用

- 物體輪廓提取
- 車道線偵測
- 文件邊緣檢測（自動裁切）
- 作為輪廓檢測的前置處理

> **提示**：Canny 對雜訊敏感，務必先做模糊處理！
''',
            'params': [
                {'name': 'threshold1', 'label': '低閾值', 'type': 'slider',
                 'min': 0, 'max': 200, 'step': 10, 'default': 50},
                {'name': 'threshold2', 'label': '高閾值', 'type': 'slider',
                 'min': 50, 'max': 300, 'step': 10, 'default': 150},
                {'name': 'blur', 'label': '先做高斯模糊', 'type': 'checkbox', 'default': True}
            ]
        },

        # ===== 金字塔 =====
        'pyramid': {
            'name': '影像金字塔',
            'category': '金字塔',
            'description': 'pyrDown 向下取樣(縮小)，每次尺寸減半。pyrUp 向上取樣(放大)。注意：縮小後再放大無法完全還原，會損失資訊。',
            'params': [
                {'name': 'operation', 'label': '操作類型', 'type': 'select',
                 'options': [
                     {'value': 'down', 'label': '向下取樣 (縮小)'},
                     {'value': 'up', 'label': '向上取樣 (放大)'},
                     {'value': 'laplacian', 'label': '拉普拉斯金字塔'}
                 ], 'default': 'down'},
                {'name': 'levels', 'label': '層數', 'type': 'slider',
                 'min': 1, 'max': 4, 'step': 1, 'default': 2}
            ]
        },

        # ===== 輪廓 =====
        'contours': {
            'name': '影像輪廓',
            'category': '輪廓',
            'description': '''尋找並繪製影像中物體的邊界，是物件檢測的核心技術。

## 基本流程

```
原始影像 → 灰階 → 二值化 → 尋找輪廓 → 繪製/分析
```

### 程式碼

```python
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 尋找輪廓
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,     # 只找外輪廓
    cv2.CHAIN_APPROX_SIMPLE # 壓縮輪廓點
)

# 繪製所有輪廓（-1 表示全部）
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
```

## 檢索模式

| 模式 | 說明 |
|------|------|
| `RETR_EXTERNAL` | 只找最外層輪廓 |
| `RETR_LIST` | 找所有輪廓（無階層） |
| `RETR_TREE` | 找所有輪廓（有階層關係） |

## 輪廓特徵

```python
for cnt in contours:
    # 面積
    area = cv2.contourArea(cnt)

    # 周長
    perimeter = cv2.arcLength(cnt, True)

    # 外接矩形
    x, y, w, h = cv2.boundingRect(cnt)

    # 最小外接矩形（可旋轉）
    rect = cv2.minAreaRect(cnt)

    # 最小外接圓
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)

    # 凸包
    hull = cv2.convexHull(cnt)

    # 輪廓近似（多邊形）
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # len(approx) == 3: 三角形
    # len(approx) == 4: 矩形
    # len(approx) > 4: 圓形
```

### 實際應用

- **物件計數**：計算細胞、硬幣數量
- **形狀辨識**：根據頂點數判斷形狀
- **手勢識別**：分析手部輪廓
- **文件掃描**：找出紙張四個角落

> **提示**：使用 `cv2.contourArea()` 過濾太小的輪廓可以去除雜訊！
''',
            'params': [
                {'name': 'mode', 'label': '檢索模式', 'type': 'select',
                 'options': [
                     {'value': 'external', 'label': '只有外輪廓'},
                     {'value': 'tree', 'label': '階層結構'}
                 ], 'default': 'external'},
                {'name': 'draw', 'label': '繪製類型', 'type': 'select',
                 'options': [
                     {'value': 'contour', 'label': '輪廓線'},
                     {'value': 'bbox', 'label': '外接矩形'},
                     {'value': 'minrect', 'label': '最小外接矩形'},
                     {'value': 'circle', 'label': '最小外接圓'},
                     {'value': 'hull', 'label': '凸包'}
                 ], 'default': 'contour'},
                {'name': 'thresh', 'label': '二值化閾值', 'type': 'slider',
                 'min': 0, 'max': 255, 'step': 5, 'default': 127}
            ]
        },

        # ===== 長條圖 =====
        'histogram': {
            'name': '長條圖處理',
            'category': '長條圖',
            'description': '''透過直方圖均衡化增強影像對比度，讓暗部和亮部細節更清晰。

## 直方圖均衡化

將像素值分布「拉開」，讓整個範圍（0-255）都被使用：

```python
import cv2

img = cv2.imread('dark_photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 直方圖均衡化
equalized = cv2.equalizeHist(gray)
```

### 彩色影像處理

不能直接對 BGR 做均衡化（會變色），要轉 YCrCb 或 LAB：

```python
# 轉換到 YCrCb
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# 只對 Y（亮度）通道做均衡化
ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])

# 轉回 BGR
result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
```

## CLAHE 自適應均衡化

**C**ontrast **L**imited **A**daptive **H**istogram **E**qualization

普通均衡化可能過度增強，CLAHE 會：
1. 將影像分成小區塊（tiles）
2. 對每個區塊分別做均衡化
3. 限制對比度（clipLimit）避免過度放大

```python
# 建立 CLAHE 物件
clahe = cv2.createCLAHE(
    clipLimit=2.0,    # 對比度限制
    tileGridSize=(8, 8)  # 區塊大小
)

# 套用 CLAHE
result = clahe.apply(gray)
```

### 參數說明

| 參數 | 說明 | 建議值 |
|------|------|--------|
| clipLimit | 對比度限制 | 2.0-4.0 |
| tileGridSize | 區塊大小 | (8,8) |

### 實際應用

- **醫學影像**：增強 X 光、CT 的細節
- **夜視影像**：提高低光照影像可見度
- **文件掃描**：增強褪色文件的對比度

> **提示**：CLAHE 比普通均衡化更自然，不會產生過度飽和的感覺！
''',
            'params': [
                {'name': 'method', 'label': '處理方法', 'type': 'select',
                 'options': [
                     {'value': 'equalize', 'label': '直方圖均衡化'},
                     {'value': 'clahe', 'label': 'CLAHE 自適應均衡化'}
                 ], 'default': 'equalize'},
                {'name': 'clip_limit', 'label': 'CLAHE 截斷限制', 'type': 'slider',
                 'min': 1.0, 'max': 10.0, 'step': 0.5, 'default': 2.0},
                {'name': 'tile_size', 'label': 'CLAHE 區塊大小', 'type': 'slider',
                 'min': 2, 'max': 16, 'step': 2, 'default': 8}
            ]
        },

        # ===== 頻域 =====
        'fourier': {
            'name': '傅立葉轉換',
            'category': '頻域',
            'description': '''將影像從空間域轉換到頻率域，揭示影像的頻率組成。

## 核心概念

- **低頻**（頻譜中心）：代表平緩變化的區域，如均勻背景
- **高頻**（頻譜邊緣）：代表快速變化的區域，如邊緣、細節

### 程式碼

```python
import cv2
import numpy as np

gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 傅立葉轉換
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)  # 將低頻移到中心

# 計算頻譜（取對數增強可視性）
magnitude = 20 * np.log(np.abs(fshift) + 1)
```

## 頻域濾波

### 低通濾波（保留低頻 → 模糊）

```python
rows, cols = gray.shape
crow, ccol = rows//2, cols//2
radius = 30

# 建立遮罩：中心圓形區域
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 1, -1)

# 濾波
fshift_filtered = fshift * mask
f_ishift = np.fft.ifftshift(fshift_filtered)
result = np.abs(np.fft.ifft2(f_ishift))
```

### 高通濾波（保留高頻 → 邊緣）

```python
# 反轉遮罩
mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), radius, 0, -1)
```

## 頻譜的意義

```
頻譜圖中心 = 影像平均亮度
頻譜圖十字 = 水平/垂直線條
頻譜圖對角線 = 斜向紋理
```

### 實際應用

- **去除週期性雜訊**：在頻譜中找出異常亮點並遮蔽
- **影像壓縮**：JPEG 使用離散餘弦轉換（DCT）
- **紋理分析**：分析布料、木紋等週期性圖案
- **濾波器設計**：設計理想的低通/高通/帶通濾波器

> **提示**：傅立葉轉換是訊號處理的基礎，理解它有助於深入學習影像處理！
''',
            'params': [
                {'name': 'display', 'label': '顯示類型', 'type': 'select',
                 'options': [
                     {'value': 'magnitude', 'label': '頻譜圖'},
                     {'value': 'lowpass', 'label': '低通濾波 (模糊)'},
                     {'value': 'highpass', 'label': '高通濾波 (邊緣)'}
                 ], 'default': 'magnitude'},
                {'name': 'radius', 'label': '濾波半徑', 'type': 'slider',
                 'min': 5, 'max': 100, 'step': 5, 'default': 30}
            ]
        },

        # ===== 霍夫 =====
        'hough_line': {
            'name': '霍夫直線轉換',
            'category': '霍夫',
            'description': '在邊緣影像中偵測直線。使用機率霍夫變換可指定最小線段長度和最大間隙，更實用。',
            'params': [
                {'name': 'threshold', 'label': '投票閾值', 'type': 'slider',
                 'min': 10, 'max': 200, 'step': 10, 'default': 50},
                {'name': 'min_length', 'label': '最小線段長度', 'type': 'slider',
                 'min': 10, 'max': 200, 'step': 10, 'default': 50},
                {'name': 'max_gap', 'label': '最大間隙', 'type': 'slider',
                 'min': 1, 'max': 50, 'step': 5, 'default': 10}
            ]
        },
        'hough_circle': {
            'name': '霍夫圓形轉換',
            'category': '霍夫',
            'description': '在影像中偵測圓形。需要先做模糊減少雜訊。可設定圓心距離、半徑範圍等參數。',
            'params': [
                {'name': 'dp', 'label': '累加器解析度', 'type': 'slider',
                 'min': 1, 'max': 3, 'step': 0.5, 'default': 1},
                {'name': 'min_dist', 'label': '圓心最小距離', 'type': 'slider',
                 'min': 10, 'max': 100, 'step': 10, 'default': 20},
                {'name': 'param1', 'label': 'Canny 高閾值', 'type': 'slider',
                 'min': 10, 'max': 200, 'step': 10, 'default': 50},
                {'name': 'param2', 'label': '累加器閾值', 'type': 'slider',
                 'min': 10, 'max': 100, 'step': 5, 'default': 30},
                {'name': 'min_radius', 'label': '最小半徑', 'type': 'slider',
                 'min': 0, 'max': 100, 'step': 5, 'default': 0},
                {'name': 'max_radius', 'label': '最大半徑 (0=無限)', 'type': 'slider',
                 'min': 0, 'max': 200, 'step': 10, 'default': 0}
            ]
        },

        # ===== 分割 =====
        'watershed': {
            'name': '分水嶺演算法',
            'category': '分割',
            'description': '將影像視為地形，從標記點開始填充，在邊界相遇處形成分水嶺。常用於分割相連的物體（如細胞、硬幣）。',
            'params': [
                {'name': 'thresh', 'label': '二值化閾值', 'type': 'slider',
                 'min': 0, 'max': 255, 'step': 5, 'default': 0},
                {'name': 'dist_thresh', 'label': '距離閾值比例', 'type': 'slider',
                 'min': 0.1, 'max': 0.9, 'step': 0.1, 'default': 0.7}
            ]
        },

        # ===== 識別 =====
        'face_detection': {
            'name': '人臉識別',
            'category': '識別',
            'description': '''使用 Haar Cascade 分類器偵測人臉，這是 OpenCV 內建的傳統機器學習方法。

## Haar Cascade 原理

使用大量「類 Haar 特徵」進行快速篩選：

```
眼睛特徵：上方較暗、下方較亮
┌─────┬─────┐
│ 黑  │ 黑  │  ← 眼眉
├─────┼─────┤
│ 白  │ 白  │  ← 眼下皮膚
└─────┴─────┘
```

### 程式碼

```python
import cv2

# 載入預訓練的分類器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

img = cv2.imread('group_photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 偵測人臉
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # 縮放比例
    minNeighbors=4,    # 最小鄰居數
    minSize=(30, 30)   # 最小人臉尺寸
)

# 繪製矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

### 參數說明

| 參數 | 說明 | 調整建議 |
|------|------|----------|
| scaleFactor | 每次縮小的比例 | 1.1-1.3，越小越精確但越慢 |
| minNeighbors | 確認偵測的鄰居數 | 3-6，越大誤報越少 |
| minSize | 最小人臉尺寸 | 視影像大小調整 |

### OpenCV 內建的其他分類器

```python
# 眼睛
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# 微笑
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)

# 貓臉
cat_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
)
```

### 與深度學習比較

| 特性 | Haar Cascade | 深度學習 (DNN) |
|------|--------------|----------------|
| 速度 | ⚡ 非常快 | 較慢 |
| 精確度 | 中 | **高** |
| 角度容忍 | 差（需正面） | 好 |
| 遮擋處理 | 差 | 較好 |

### 實際應用

- 相機自動對焦
- 影片中的人臉追蹤
- 簡單的人數統計
- 配合其他特徵做表情識別

> **提示**：Haar Cascade 對正面臉效果最好，側臉可能漏偵測！
''',
            'params': [
                {'name': 'scale_factor', 'label': '縮放比例', 'type': 'slider',
                 'min': 1.05, 'max': 1.5, 'step': 0.05, 'default': 1.1},
                {'name': 'min_neighbors', 'label': '最小鄰居數', 'type': 'slider',
                 'min': 1, 'max': 10, 'step': 1, 'default': 4},
                {'name': 'min_size', 'label': '最小人臉大小', 'type': 'slider',
                 'min': 20, 'max': 100, 'step': 10, 'default': 30}
            ]
        },
    }


def process_image(img, effect, params=None):
    """處理影像並返回結果和程式碼"""
    if params is None:
        params = {}

    processors = {
        'original': process_original,
        'channels': process_channels,
        'arithmetic': process_arithmetic,
        'weighted': process_weighted,
        'bitwise': process_bitwise,
        'bit_plane': process_bit_plane,
        'encrypt': process_encrypt,
        'color_space': process_color_space,
        'geometric': process_geometric,
        'threshold': process_threshold,
        'blur_mean': process_blur_mean,
        'blur_gaussian': process_blur_gaussian,
        'blur_median': process_blur_median,
        'blur_bilateral': process_blur_bilateral,
        'morphology': process_morphology,
        'gradient_sobel': process_gradient_sobel,
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


# ===== 處理函數 =====

def process_original(img, params):
    code = '''import cv2

# 讀取影像
img = cv2.imread('image.jpg')

# 顯示影像資訊
print(f'影像大小: {img.shape}')
print(f'資料類型: {img.dtype}')

# 顯示影像
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
    return img, code


def process_channels(img, params):
    channel = params.get('channel', 'all')
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(b)

    if channel == 'b':
        result = cv2.merge([b, zeros, zeros])
    elif channel == 'g':
        result = cv2.merge([zeros, g, zeros])
    elif channel == 'r':
        result = cv2.merge([zeros, zeros, r])
    else:
        h, w = img.shape[:2]
        scale = 0.33
        b_img = cv2.resize(cv2.merge([b, zeros, zeros]), (int(w*scale), int(h*scale)))
        g_img = cv2.resize(cv2.merge([zeros, g, zeros]), (int(w*scale), int(h*scale)))
        r_img = cv2.resize(cv2.merge([zeros, zeros, r]), (int(w*scale), int(h*scale)))
        result = np.hstack([b_img, g_img, r_img])

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 拆解 BGR 三個通道
b, g, r = cv2.split(img)

# 目前選擇: {channel}
zeros = np.zeros_like(b)

# 建立單獨顯示各通道的彩色圖
b_img = cv2.merge([b, zeros, zeros])  # 藍色通道
g_img = cv2.merge([zeros, g, zeros])  # 綠色通道
r_img = cv2.merge([zeros, zeros, r])  # 紅色通道

# 合併通道
merged = cv2.merge([b, g, r])'''
    return result, code


def process_arithmetic(img, params):
    operation = params.get('operation', 'add')
    value = int(params.get('value', 50))

    if operation == 'add':
        result = cv2.add(img, np.ones_like(img) * value)
    else:
        result = cv2.subtract(img, np.ones_like(img) * value)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 操作: {operation}, 調整值: {value}

# 加法 (變亮) - cv2.add 會處理溢位
bright = cv2.add(img, np.ones_like(img) * {value})

# 減法 (變暗)
dark = cv2.subtract(img, np.ones_like(img) * {value})

# 注意差異:
# cv2.add(img, 50) -> 超過 255 會變成 255
# img + 50 -> 超過 255 會溢位 (256 變成 0)'''
    return result, code


def process_weighted(img, params):
    alpha = float(params.get('alpha', 0.7))
    blend_type = params.get('blend_type', 'gradient_h')
    color_r = int(params.get('color_r', 255))
    color_g = int(params.get('color_g', 255))
    color_b = int(params.get('color_b', 255))
    h, w = img.shape[:2]

    # 生成第二張圖
    blend_img = np.zeros_like(img)

    if blend_type == 'gradient_h':
        # 水平漸層
        for i in range(w):
            ratio = i / w
            blend_img[:, i] = [int(color_b * ratio), int(color_g * ratio), int(color_r * ratio)]
    elif blend_type == 'gradient_v':
        # 垂直漸層
        for j in range(h):
            ratio = j / h
            blend_img[j, :] = [int(color_b * ratio), int(color_g * ratio), int(color_r * ratio)]
    elif blend_type == 'gradient_d':
        # 對角漸層
        for j in range(h):
            for i in range(w):
                ratio = (i + j) / (w + h)
                blend_img[j, i] = [int(color_b * ratio), int(color_g * ratio), int(color_r * ratio)]
    elif blend_type == 'radial':
        # 放射漸層
        cx, cy = w // 2, h // 2
        max_dist = np.sqrt(cx**2 + cy**2)
        for j in range(h):
            for i in range(w):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                ratio = dist / max_dist
                blend_img[j, i] = [int(color_b * ratio), int(color_g * ratio), int(color_r * ratio)]
    elif blend_type == 'solid':
        # 純色
        blend_img[:] = [color_b, color_g, color_r]
    elif blend_type == 'checker':
        # 棋盤格
        block_size = max(w, h) // 8
        for j in range(h):
            for i in range(w):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    blend_img[j, i] = [color_b, color_g, color_r]
    elif blend_type == 'noise':
        # 隨機雜訊
        blend_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    result = cv2.addWeighted(img, alpha, blend_img, 1-alpha, 0)

    blend_desc = {
        'gradient_h': '水平漸層', 'gradient_v': '垂直漸層', 'gradient_d': '對角漸層',
        'radial': '放射漸層', 'solid': '純色', 'checker': '棋盤格', 'noise': '隨機雜訊'
    }

    code = f'''import cv2
import numpy as np

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')  # 或自行生成

# 混合類型: {blend_desc.get(blend_type, blend_type)}

# 範例: 生成水平漸層圖
h, w = img1.shape[:2]
gradient = np.zeros_like(img1)
for i in range(w):
    ratio = i / w
    gradient[:, i] = [int(255 * ratio)] * 3

# 影像加權和公式: dst = src1 × α + src2 × β + γ
alpha = {alpha}  # 原圖權重
beta = {1-alpha:.2f}   # 混合圖權重
gamma = 0        # 亮度偏移

result = cv2.addWeighted(img1, alpha, gradient, beta, gamma)

# α 越大，原圖越清楚
# β 越大，混合圖越明顯
# α + β = 1 時，整體亮度不變'''
    return result, code


def process_bitwise(img, params):
    operation = params.get('operation', 'and')
    shape = params.get('shape', 'circle')
    mask_size = int(params.get('mask_size', 50))
    center_x = int(params.get('center_x', 50))
    center_y = int(params.get('center_y', 50))
    show_mask = params.get('show_mask', False)
    h, w = img.shape[:2]

    # 計算遮罩位置和大小
    cx = int(w * center_x / 100)
    cy = int(h * center_y / 100)
    size = int(min(h, w) * mask_size / 100 / 2)

    # 建立遮罩
    mask = np.zeros((h, w), dtype=np.uint8)
    if shape == 'circle':
        cv2.circle(mask, (cx, cy), size, 255, -1)
    elif shape == 'rectangle':
        cv2.rectangle(mask, (cx - size, cy - size), (cx + size, cy + size), 255, -1)
    elif shape == 'ellipse':
        cv2.ellipse(mask, (cx, cy), (size, size // 2), 0, 0, 360, 255, -1)
    elif shape == 'triangle':
        pts = np.array([
            [cx, cy - size],
            [cx - size, cy + size],
            [cx + size, cy + size]
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # 如果只顯示遮罩
    if show_mask:
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif operation == 'and':
        result = cv2.bitwise_and(img, img, mask=mask)
    elif operation == 'or':
        mask_3ch = cv2.merge([mask, mask, mask])
        result = cv2.bitwise_or(img, mask_3ch)
    elif operation == 'xor':
        mask_3ch = cv2.merge([mask, mask, mask])
        result = cv2.bitwise_xor(img, mask_3ch)
    else:  # not
        result = cv2.bitwise_not(img)

    shape_code = {
        'circle': f'cv2.circle(mask, ({cx}, {cy}), {size}, 255, -1)',
        'rectangle': f'cv2.rectangle(mask, ({cx - size}, {cy - size}), ({cx + size}, {cy + size}), 255, -1)',
        'ellipse': f'cv2.ellipse(mask, ({cx}, {cy}), ({size}, {size // 2}), 0, 0, 360, 255, -1)',
        'triangle': 'pts = np.array([...], np.int32); cv2.fillPoly(mask, [pts], 255)'
    }

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 建立遮罩 (形狀: {shape}, 大小: {mask_size}%, 位置: ({center_x}%, {center_y}%))
mask = np.zeros((h, w), dtype=np.uint8)
{shape_code.get(shape, '')}

# 逐位元運算說明：
# 像素值 200 = 二進位 11001000
# 像素值  50 = 二進位 00110010
# AND 結果   = 二進位 00000000 = 0  (兩者都為1才是1)
# OR 結果    = 二進位 11111010 = 250 (任一為1就是1)
# XOR 結果   = 二進位 11111010 = 250 (不同才是1)

# 執行 {operation.upper()} 運算
{"result = cv2.bitwise_and(img, img, mask=mask)  # 用 mask 提取區域" if operation == 'and' else ""}
{"mask_3ch = cv2.merge([mask, mask, mask]); result = cv2.bitwise_or(img, mask_3ch)" if operation == 'or' else ""}
{"mask_3ch = cv2.merge([mask, mask, mask]); result = cv2.bitwise_xor(img, mask_3ch)" if operation == 'xor' else ""}
{"result = cv2.bitwise_not(img)  # 反轉所有像素" if operation == 'not' else ""}

# 實際應用：
# AND + 遮罩：去背、提取特定區域
# XOR：找出兩張圖的差異、簡易加密'''
    return result, code


def process_bit_plane(img, params):
    plane = int(params.get('plane', 7))
    show_all = params.get('show_all', False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if show_all:
        # 顯示所有 8 個位元平面
        h, w = gray.shape
        # 2x4 排列
        cell_h, cell_w = h // 2, w // 4
        result = np.zeros((h, w), dtype=np.uint8)

        for i in range(8):
            row, col = i // 4, i % 4
            bit_plane = ((gray >> (7 - i)) & 1) * 255
            resized = cv2.resize(bit_plane, (cell_w, cell_h))
            result[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = resized

        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        # 加上標籤
        for i in range(8):
            row, col = i // 4, i % 4
            cv2.putText(result, f'Bit {7-i}', (col*cell_w + 5, row*cell_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        bit_plane = (gray >> plane) & 1
        result = (bit_plane * 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 像素值的二進位表示
# 例如 200 = 11001000 (二進位)
#          ^^^^^^^^
#          76543210  <- 位元位置

# 提取第 {plane} 個位元平面 (0=LSB 最低位, 7=MSB 最高位)
bit_plane = (img >> {plane}) & 1  # 右移後取最低位
bit_plane = bit_plane * 255       # 0/1 -> 0/255

# 各位元平面的意義：
# Bit 7 (MSB): 代表 128，包含最主要的結構資訊
# Bit 6: 代表 64
# Bit 5: 代表 32
# Bit 4: 代表 16
# ...
# Bit 0 (LSB): 代表 1，幾乎是隨機雜訊

# 影像隱寫術範例：在 LSB 藏入訊息
secret = 1  # 要藏的位元
img_with_secret = (img & 0xFE) | secret  # 清除 LSB 後設定新值

# 重建影像（只保留高位元）
reconstructed = (img >> 4) << 4  # 只保留高 4 位元，減少 16 倍資料量'''
    return result, code


def process_encrypt(img, params):
    seed = int(params.get('seed', 42))
    display = params.get('display', 'both')
    h, w = img.shape[:2]

    # 加密
    np.random.seed(seed)
    key = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    encrypted = cv2.bitwise_xor(img, key)

    # 解密
    np.random.seed(seed)
    key = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    decrypted = cv2.bitwise_xor(encrypted, key)

    if display == 'encrypted':
        result = encrypted
    elif display == 'decrypted':
        result = decrypted
    else:  # both
        # 並排顯示：加密 | 解密
        result = np.hstack([encrypted, decrypted])
        # 加上標籤
        cv2.putText(result, 'Encrypted', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result, 'Decrypted', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# === 加密過程 ===
# 1. 設定金鑰種子（必須記住這個數字才能解密！）
np.random.seed({seed})

# 2. 產生與圖片同大小的隨機金鑰
key = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

# 3. 用 XOR 加密
encrypted = cv2.bitwise_xor(img, key)
cv2.imwrite('encrypted.png', encrypted)

# === 解密過程 ===
# 1. 用「相同的」種子重新產生金鑰
np.random.seed({seed})  # 必須與加密時相同！
key = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

# 2. 再次 XOR 就能還原
decrypted = cv2.bitwise_xor(encrypted, key)

# === XOR 還原原理 ===
# 原始像素:   11001000 (200)
# 金鑰:       00110010 (50)
# 加密結果:   11111010 (250) = 200 XOR 50
# 解密:       11111010 XOR 00110010 = 11001000 (200) 還原了！'''
    return result, code


def process_color_space(img, params):
    space = params.get('space', 'hsv_h')
    colorize_h = params.get('colorize_h', True)

    if space == 'gray':
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif space == 'rgb':
        result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif space == 'hsv':
        # 顯示 HSV 的效果（直接轉回 BGR 會有奇怪的顏色，這是預期的）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = hsv.copy()
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    elif space == 'hsv_h':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        if colorize_h:
            # 用彩色方式顯示 H 通道（色相環）
            colored_h = np.zeros_like(img)
            colored_h[:, :, 0] = h_channel  # H
            colored_h[:, :, 1] = 255  # S = 255 (全飽和)
            colored_h[:, :, 2] = 255  # V = 255 (全亮)
            result = cv2.cvtColor(colored_h, cv2.COLOR_HSV2BGR)
        else:
            result = cv2.cvtColor(h_channel, cv2.COLOR_GRAY2BGR)
    elif space == 'hsv_s':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)
    elif space == 'hsv_v':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)
    elif space == 'hls':
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        result = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    elif space == 'hls_h':
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h_channel = hls[:, :, 0]
        if colorize_h:
            colored_h = np.zeros_like(img)
            colored_h[:, :, 0] = h_channel
            colored_h[:, :, 1] = 128  # L
            colored_h[:, :, 2] = 255  # S
            result = cv2.cvtColor(colored_h, cv2.COLOR_HLS2BGR)
        else:
            result = cv2.cvtColor(h_channel, cv2.COLOR_GRAY2BGR)
    elif space == 'hls_l':
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        result = cv2.cvtColor(hls[:, :, 1], cv2.COLOR_GRAY2BGR)
    elif space == 'hls_s':
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        result = cv2.cvtColor(hls[:, :, 2], cv2.COLOR_GRAY2BGR)
    elif space == 'lab':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif space == 'lab_l':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        result = cv2.cvtColor(lab[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif space == 'lab_a':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        result = cv2.cvtColor(lab[:, :, 1], cv2.COLOR_GRAY2BGR)
    elif space == 'lab_b':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        result = cv2.cvtColor(lab[:, :, 2], cv2.COLOR_GRAY2BGR)
    elif space == 'ycrcb':
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif space == 'ycrcb_y':
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        result = cv2.cvtColor(ycrcb[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif space == 'ycrcb_cr':
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        result = cv2.cvtColor(ycrcb[:, :, 1], cv2.COLOR_GRAY2BGR)
    elif space == 'ycrcb_cb':
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        result = cv2.cvtColor(ycrcb[:, :, 2], cv2.COLOR_GRAY2BGR)
    else:
        result = img

    space_desc = {
        'gray': '灰階', 'rgb': 'RGB', 'hsv': 'HSV', 'hsv_h': 'HSV-H色相',
        'hsv_s': 'HSV-S飽和度', 'hsv_v': 'HSV-V明度',
        'hls': 'HLS', 'hls_h': 'HLS-H色相', 'hls_l': 'HLS-L亮度', 'hls_s': 'HLS-S飽和度',
        'lab': 'LAB', 'lab_l': 'LAB-L亮度', 'lab_a': 'LAB-A', 'lab_b': 'LAB-B',
        'ycrcb': 'YCrCb', 'ycrcb_y': 'YCrCb-Y亮度', 'ycrcb_cr': 'YCrCb-Cr', 'ycrcb_cb': 'YCrCb-Cb'
    }

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 色彩空間: {space_desc.get(space, space)}

# === 各色彩空間說明 ===

# 灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# HSV (色相 H, 飽和度 S, 明度 V)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
# H: 0-180 (OpenCV 用 180 而非 360，紅色約 0/180，綠色約 60，藍色約 120)
# S: 0-255 (越高越鮮豔)
# V: 0-255 (越高越亮)

# 用 HSV 過濾特定顏色（例如：找紅色物體）
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

# HLS (色相 H, 亮度 L, 飽和度 S)
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# LAB (接近人眼感知)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# L: 亮度 0-255
# A: 綠(-128)到紅(+127)
# B: 藍(-128)到黃(+127)

# YCrCb (JPEG/影片常用)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# Y: 亮度
# Cr: 紅色色度
# Cb: 藍色色度

# 膚色偵測範例 (YCrCb)
lower_skin = np.array([0, 133, 77])
upper_skin = np.array([255, 173, 127])
skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)'''
    return result, code


def process_geometric(img, params):
    transform = params.get('transform', 'rotate')
    angle = float(params.get('angle', 45))
    scale = float(params.get('scale', 1.0))
    h, w = img.shape[:2]

    if transform == 'rotate':
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(img, M, (w, h))
    elif transform == 'scale':
        result = cv2.resize(img, None, fx=scale, fy=scale)
    elif transform == 'flip_h':
        result = cv2.flip(img, 1)
    elif transform == 'flip_v':
        result = cv2.flip(img, 0)
    elif transform == 'flip_both':
        result = cv2.flip(img, -1)
    else:
        result = img

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 轉換類型: {transform}

# 旋轉 (角度: {angle})
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, {angle}, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# 縮放 (比例: {scale})
scaled = cv2.resize(img, None, fx={scale}, fy={scale})

# 翻轉
flip_h = cv2.flip(img, 1)    # 水平翻轉
flip_v = cv2.flip(img, 0)    # 垂直翻轉
flip_both = cv2.flip(img, -1)  # 雙向翻轉'''
    return result, code


def process_threshold(img, params):
    method = params.get('method', 'otsu')
    thresh = int(params.get('thresh', 127))
    block_size = int(params.get('block_size', 11))
    if block_size % 2 == 0:
        block_size += 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == 'binary':
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    elif method == 'binary_inv':
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    elif method == 'otsu':
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_mean':
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, block_size, 2)
    elif method == 'adaptive_gaussian':
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, 2)
    else:
        result = gray

    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = f'''import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 二值化方法: {method}

# 固定閾值 (閾值: {thresh})
_, binary = cv2.threshold(img, {thresh}, 255, cv2.THRESH_BINARY)

# Otsu 自動閾值
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 自適應閾值 (區塊大小: {block_size})
adaptive = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    {block_size},  # 區塊大小 (必須是奇數)
    2              # 減去的常數
)'''
    return result, code


def process_blur_mean(img, params):
    ksize = int(params.get('ksize', 5))
    if ksize % 2 == 0:
        ksize += 1

    result = cv2.blur(img, (ksize, ksize))

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 均值濾波 (核心大小: {ksize}x{ksize})
blurred = cv2.blur(img, ({ksize}, {ksize}))

# 等同於自定義卷積核
kernel = np.ones(({ksize}, {ksize}), np.float32) / ({ksize} * {ksize})
blurred = cv2.filter2D(img, -1, kernel)

# 卷積概念:
# 將核心滑過影像，計算加權平均值
# 核心越大，模糊效果越強'''
    return result, code


def process_blur_gaussian(img, params):
    ksize = int(params.get('ksize', 5))
    sigma = float(params.get('sigma', 0))
    if ksize % 2 == 0:
        ksize += 1

    result = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    code = f'''import cv2

img = cv2.imread('image.jpg')

# 高斯濾波 (核心: {ksize}x{ksize}, sigma: {sigma})
blurred = cv2.GaussianBlur(img, ({ksize}, {ksize}), {sigma})

# sigma = 0 時自動計算: sigma = 0.3 * ((ksize-1) * 0.5 - 1) + 0.8

# 高斯濾波特點:
# - 權重呈高斯分布，中心權重最大
# - 比均值濾波更自然
# - 適合去除高斯雜訊'''
    return result, code


def process_blur_median(img, params):
    ksize = int(params.get('ksize', 5))
    if ksize % 2 == 0:
        ksize += 1

    result = cv2.medianBlur(img, ksize)

    code = f'''import cv2

img = cv2.imread('image.jpg')

# 中值濾波 (核心大小: {ksize})
blurred = cv2.medianBlur(img, {ksize})

# 中值濾波特點:
# - 使用中位數，非線性濾波
# - 對椒鹽雜訊特別有效
# - 能保持邊緣清晰
# - 運算速度較慢'''
    return result, code


def process_blur_bilateral(img, params):
    d = int(params.get('d', 9))
    sigma_color = float(params.get('sigma_color', 75))
    sigma_space = float(params.get('sigma_space', 75))

    result = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    code = f'''import cv2

img = cv2.imread('image.jpg')

# 雙邊濾波
# d: 鄰域直徑 = {d}
# sigmaColor: 顏色空間標準差 = {sigma_color}
# sigmaSpace: 座標空間標準差 = {sigma_space}
blurred = cv2.bilateralFilter(img, {d}, {sigma_color}, {sigma_space})

# 雙邊濾波特點:
# - 同時考慮空間距離和顏色相似度
# - 邊緣保持濾波 (只模糊相似顏色區域)
# - 常用於美顏磨皮效果
# - 運算速度最慢'''
    return result, code


def process_morphology(img, params):
    operation = params.get('operation', 'erosion')
    kernel_size = int(params.get('kernel_size', 5))
    kernel_shape = params.get('kernel_shape', 'rect')
    iterations = int(params.get('iterations', 1))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    shape_map = {'rect': cv2.MORPH_RECT, 'ellipse': cv2.MORPH_ELLIPSE, 'cross': cv2.MORPH_CROSS}
    kernel = cv2.getStructuringElement(shape_map[kernel_shape], (kernel_size, kernel_size))

    op_map = {
        'erosion': lambda x: cv2.erode(x, kernel, iterations=iterations),
        'dilation': lambda x: cv2.dilate(x, kernel, iterations=iterations),
        'opening': lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel),
        'closing': lambda x: cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel),
        'gradient': lambda x: cv2.morphologyEx(x, cv2.MORPH_GRADIENT, kernel),
        'tophat': lambda x: cv2.morphologyEx(x, cv2.MORPH_TOPHAT, kernel),
        'blackhat': lambda x: cv2.morphologyEx(x, cv2.MORPH_BLACKHAT, kernel),
    }

    result = op_map[operation](binary)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = f'''import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 建立結構元素 (形狀: {kernel_shape}, 大小: {kernel_size}x{kernel_size})
kernel = cv2.getStructuringElement(cv2.MORPH_{kernel_shape.upper()}, ({kernel_size}, {kernel_size}))

# 形態學操作: {operation}
# 侵蝕: 縮小白色區域
erosion = cv2.erode(binary, kernel, iterations={iterations})
# 膨脹: 擴大白色區域
dilation = cv2.dilate(binary, kernel, iterations={iterations})
# 開運算 = 侵蝕 + 膨脹 (去除小白點)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# 閉運算 = 膨脹 + 侵蝕 (填補小黑洞)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
# 梯度 = 膨脹 - 侵蝕 (邊緣)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)'''
    return result, code


def process_gradient_sobel(img, params):
    direction = params.get('direction', 'both')
    ksize = int(params.get('ksize', '3'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if direction == 'x':
        result = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        result = cv2.convertScaleAbs(result)
    elif direction == 'y':
        result = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        result = cv2.convertScaleAbs(result)
    else:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        result = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5,
                                 cv2.convertScaleAbs(sobely), 0.5, 0)

    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = f'''import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel 梯度 (方向: {direction}, 核心: {ksize})
# X 方向 (檢測垂直邊緣)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize={ksize})
# Y 方向 (檢測水平邊緣)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize={ksize})

# 轉換為可顯示格式
abs_sobelx = cv2.convertScaleAbs(sobelx)
abs_sobely = cv2.convertScaleAbs(sobely)

# 合併 X 和 Y 方向
sobel = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)'''
    return result, code


def process_gradient_laplacian(img, params):
    ksize = int(params.get('ksize', '3'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    result = cv2.convertScaleAbs(laplacian)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = f'''import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Laplacian 二階導數 (核心: {ksize})
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize={ksize})
laplacian = cv2.convertScaleAbs(laplacian)

# Laplacian 特點:
# - 檢測所有方向的邊緣
# - 對雜訊敏感 (建議先做高斯模糊)
# - 可用於影像銳化: sharpened = img - laplacian'''
    return result, code


def process_canny(img, params):
    threshold1 = int(params.get('threshold1', 50))
    threshold2 = int(params.get('threshold2', 150))
    do_blur = params.get('blur', True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if do_blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    result = cv2.Canny(gray, threshold1, threshold2)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = f'''import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# {'先做高斯模糊減少雜訊' if do_blur else '不做模糊'}
{'blurred = cv2.GaussianBlur(img, (5, 5), 0)' if do_blur else '# blurred = img'}

# Canny 邊緣檢測
# 低閾值: {threshold1}, 高閾值: {threshold2}
edges = cv2.Canny({'blurred' if do_blur else 'img'}, {threshold1}, {threshold2})

# 建議閾值比例: 高/低 = 2:1 或 3:1

# 自動計算閾值 (使用中位數)
import numpy as np
median = np.median(img)
low = int(max(0, 0.7 * median))
high = int(min(255, 1.3 * median))'''
    return result, code


def process_pyramid(img, params):
    operation = params.get('operation', 'down')
    levels = int(params.get('levels', 2))

    if operation == 'down':
        result = img.copy()
        for _ in range(levels):
            result = cv2.pyrDown(result)
    elif operation == 'up':
        result = img.copy()
        for _ in range(levels):
            result = cv2.pyrUp(result)
    else:  # laplacian
        down = cv2.pyrDown(img)
        up = cv2.pyrUp(down, dstsize=(img.shape[1], img.shape[0]))
        result = cv2.subtract(img, up)

    code = f'''import cv2

img = cv2.imread('image.jpg')

# 影像金字塔 (操作: {operation}, 層數: {levels})

# 向下取樣 (縮小，每次尺寸減半)
down = img.copy()
for i in range({levels}):
    down = cv2.pyrDown(down)

# 向上取樣 (放大，每次尺寸加倍)
up = img.copy()
for i in range({levels}):
    up = cv2.pyrUp(up)

# 拉普拉斯金字塔 (用於影像融合)
# L = G - pyrUp(pyrDown(G))
laplacian = img - cv2.pyrUp(cv2.pyrDown(img))'''
    return result, code


def process_contours(img, params):
    mode = params.get('mode', 'external')
    draw = params.get('draw', 'contour')
    thresh = int(params.get('thresh', 127))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    mode_map = {'external': cv2.RETR_EXTERNAL, 'tree': cv2.RETR_TREE}
    contours, _ = cv2.findContours(binary, mode_map[mode], cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        if draw == 'contour':
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
        elif draw == 'bbox':
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        elif draw == 'minrect':
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(result, [box], 0, (0, 255, 0), 2)
        elif draw == 'circle':
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(result, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        elif draw == 'hull':
            hull = cv2.convexHull(cnt)
            cv2.drawContours(result, [hull], -1, (0, 255, 0), 2)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, {thresh}, 255, cv2.THRESH_BINARY)

# 尋找輪廓 (模式: {mode})
contours, hierarchy = cv2.findContours(
    binary, cv2.RETR_{'EXTERNAL' if mode == 'external' else 'TREE'}, cv2.CHAIN_APPROX_SIMPLE
)

# 繪製: {draw}
for cnt in contours:
    area = cv2.contourArea(cnt)     # 面積
    perimeter = cv2.arcLength(cnt, True)  # 周長

    # 外接矩形
    x, y, w, h = cv2.boundingRect(cnt)

    # 最小外接矩形 (可旋轉)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    # 最小外接圓
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)

    # 凸包
    hull = cv2.convexHull(cnt)'''
    return result, code


def process_histogram(img, params):
    method = params.get('method', 'equalize')
    clip_limit = float(params.get('clip_limit', 2.0))
    tile_size = int(params.get('tile_size', 8))

    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        if method == 'equalize':
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        else:  # clahe
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])

        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        if method == 'equalize':
            result = cv2.equalizeHist(img)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            result = clahe.apply(img)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = f'''import cv2

img = cv2.imread('image.jpg')

# 方法: {method}

# 轉換到 YCrCb，只對 Y 通道處理 (保持顏色)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# 直方圖均衡化
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

# CLAHE (對比度受限自適應直方圖均衡化)
clahe = cv2.createCLAHE(
    clipLimit={clip_limit},          # 對比度限制
    tileGridSize=({tile_size}, {tile_size})  # 區塊大小
)
ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])

result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)'''
    return result, code


def process_fourier(img, params):
    display = params.get('display', 'magnitude')
    radius = int(params.get('radius', 30))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    if display == 'magnitude':
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude_spectrum = 20 * np.log(magnitude + 1)
        result = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    elif display == 'lowpass':
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, (1, 1), -1)
        filtered = dft_shift * mask
        f_ishift = np.fft.ifftshift(filtered)
        img_back = cv2.idft(f_ishift)
        result = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    else:  # highpass
        mask = np.ones((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, (0, 0), -1)
        filtered = dft_shift * mask
        f_ishift = np.fft.ifftshift(filtered)
        img_back = cv2.idft(f_ishift)
        result = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_GRAY2BGR)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 離散傅立葉轉換
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # 將低頻移到中心

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# 顯示類型: {display}
# 半徑: {radius}

# 頻譜圖
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum = 20 * np.log(magnitude + 1)

# 低通濾波 (模糊)
mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), {radius}, (1, 1), -1)

# 高通濾波 (邊緣)
mask = np.ones((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), {radius}, (0, 0), -1)

# 套用濾波並逆轉換
filtered = dft_shift * mask
f_ishift = np.fft.ifftshift(filtered)
img_back = cv2.idft(f_ishift)'''
    return result, code


def process_hough_line(img, params):
    threshold = int(params.get('threshold', 50))
    min_length = int(params.get('min_length', 50))
    max_gap = int(params.get('max_gap', 10))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    result = img.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold,
                            minLineLength=min_length, maxLineGap=max_gap)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# 機率霍夫變換
lines = cv2.HoughLinesP(
    edges,
    rho=1,                # 距離解析度
    theta=np.pi/180,      # 角度解析度
    threshold={threshold},          # 投票閾值
    minLineLength={min_length},     # 最小線段長度
    maxLineGap={max_gap}            # 最大間隙
)

# 繪製直線
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)'''
    return result, code


def process_hough_circle(img, params):
    dp = float(params.get('dp', 1))
    min_dist = int(params.get('min_dist', 20))
    param1 = int(params.get('param1', 50))
    param2 = int(params.get('param2', 30))
    min_radius = int(params.get('min_radius', 0))
    max_radius = int(params.get('max_radius', 0))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    result = img.copy()
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)  # 減少雜訊

# 霍夫圓形轉換
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp={dp},              # 累加器解析度
    minDist={min_dist},          # 圓心最小距離
    param1={param1},           # Canny 高閾值
    param2={param2},           # 累加器閾值
    minRadius={min_radius},        # 最小半徑
    maxRadius={max_radius}         # 最大半徑 (0=無限)
)

# 繪製圓形
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 圓周
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)     # 圓心'''
    return result, code


def process_watershed(img, params):
    thresh = int(params.get('thresh', 0))
    dist_thresh = float(params.get('dist_thresh', 0.7))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if thresh == 0:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, dist_thresh * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    result = img.copy()
    markers = cv2.watershed(result, markers)
    result[markers == -1] = [0, 0, 255]

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化 (閾值: {thresh if thresh > 0 else 'Otsu'})
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 開運算去除雜訊
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# 確定背景區域
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 確定前景區域 (距離轉換)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, {dist_thresh} * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# 未知區域
unknown = cv2.subtract(sure_bg, sure_fg)

# 標記連通區域
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 應用分水嶺演算法
markers = cv2.watershed(img, markers)
img[markers == -1] = [0, 0, 255]  # 邊界標記為紅色'''
    return result, code


def process_face_detection(img, params):
    scale_factor = float(params.get('scale_factor', 1.1))
    min_neighbors = int(params.get('min_neighbors', 4))
    min_size = int(params.get('min_size', 30))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_file = os.path.join(CASCADE_PATH, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_file):
        cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(cascade_file)
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors,
                                          minSize=(min_size, min_size))

    result = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    code = f'''import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 載入 Haar Cascade 分類器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 偵測人臉
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor={scale_factor},    # 每次縮放比例
    minNeighbors={min_neighbors},        # 最小鄰居數 (越高越嚴格)
    minSize=({min_size}, {min_size})       # 最小人臉大小
)

# 繪製矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 偵測到 {{len(faces)}} 張人臉'''
    return result, code
