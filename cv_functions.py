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
            'description': '顯示原始上傳的影像，不做任何處理。這是 OpenCV 讀取影像的基本操作。',
            'params': []
        },
        'channels': {
            'name': '通道拆解 (BGR)',
            'category': '基礎',
            'description': '將彩色影像拆解為藍(B)、綠(G)、紅(R)三個通道。OpenCV 使用 BGR 順序而非 RGB。每個通道都是一個灰階影像，數值範圍 0-255。',
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
            'description': '對影像進行加法或減法運算，可用於調整亮度。cv2.add() 會自動處理溢位（超過255變成255），而直接使用 + 運算子可能會溢位。',
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
            'description': '將兩張影像按比例混合：dst = src1 × α + src2 × β + γ。這個公式讓你可以控制兩張圖的混合比例。α+β 不一定要等於 1，但若總和為 1 可保持亮度。常見應用包括：影像融合（如全景圖拼接的過渡區域）、半透明浮水印、影片淡入淡出效果、將圖片疊加到背景上。γ 參數可以調整整體亮度。',
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
            'description': '「逐位元」是指對每個像素的二進位值逐一進行邏輯運算。例如像素值 200（二進位 11001000）與 50（00110010）進行 AND 運算，會得到 00000000（0）。這是因為 AND 運算只有兩個位元都是 1 時結果才是 1。\n\n【四種運算】\n• AND（交集）：兩者都為 1 才是 1，常用於「遮罩提取」，只保留遮罩為白色（255）的區域\n• OR（聯集）：任一為 1 就是 1，用於合併兩張圖的亮部\n• XOR（互斥或）：兩者不同才是 1，用於找出差異區域或簡易加密\n• NOT（反轉）：0 變 1、1 變 0，產生負片效果\n\n【實際應用】：去背合成、Logo 浮水印、遮罩選取、影像加密',
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
            'description': '每個灰階像素值（0-255）可以用 8 個二進位位元表示，例如 200 = 11001000。「位元平面分解」就是把每個位元單獨提取出來顯示。\n\n【8個平面的意義】\n• 平面 7（MSB，最高位元）：代表 128 的位置，包含影像最主要的結構資訊\n• 平面 6-4：代表 64、32、16，仍包含可辨識的影像輪廓\n• 平面 3-1：代表 8、4、2，開始變得雜亂\n• 平面 0（LSB，最低位元）：代表 1，幾乎是隨機雜訊\n\n【實際應用】\n• 影像隱寫術（Steganography）：在 LSB 藏入秘密訊息，肉眼看不出差異\n• 影像壓縮：只保留高位元平面來減少資料量\n• 浮水印：在較高位元平面嵌入標記\n• 影像分析：檢查不同位元層的資訊分布',
            'params': [
                {'name': 'plane', 'label': '位元平面 (0=LSB, 7=MSB)', 'type': 'slider',
                 'min': 0, 'max': 7, 'step': 1, 'default': 7},
                {'name': 'show_all', 'label': '顯示所有平面', 'type': 'checkbox', 'default': False}
            ]
        },
        'encrypt': {
            'name': '影像加密與解密',
            'category': '基礎',
            'description': '利用 XOR（互斥或）運算進行簡單的對稱式加密。\n\n【XOR 加密原理】\nXOR 有個神奇的特性：對同一個數值做兩次 XOR 會還原！\n• 原始像素 ⊕ 金鑰 = 加密像素\n• 加密像素 ⊕ 金鑰 = 原始像素\n\n例如：200 ⊕ 50 = 254（加密），254 ⊕ 50 = 200（解密）\n\n【金鑰種子】\n我們用隨機數產生器製造金鑰圖。只要種子相同，產生的金鑰就相同，因此：\n• 加密時：設定種子 → 產生金鑰 → XOR\n• 解密時：設定「同樣的」種子 → 產生「同樣的」金鑰 → XOR 還原\n\n【安全性說明】\n這是教學用的簡易加密，真正的影像加密會使用 AES 等專業演算法。',
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
            'description': '不同的色彩空間用不同方式描述顏色，各有其適用場景：\n\n【BGR/RGB】OpenCV 預設使用 BGR（藍綠紅）順序，一般圖片軟體用 RGB。\n\n【HSV】將顏色拆成「色相 H」（顏色種類 0-180）、「飽和度 S」（鮮豔程度）、「明度 V」（亮暗）。非常適合「顏色過濾」，例如找出畫面中所有紅色物體，只需指定 H 的範圍。\n\n【HLS】類似 HSV，但用「亮度 L」取代明度，有些場景更直觀。\n\n【LAB】L 是亮度，A 是綠-紅軸，B 是藍-黃軸。設計上接近人眼感知，常用於色彩校正、計算兩個顏色的「視覺差異」。\n\n【YCrCb】Y 是亮度，Cr/Cb 是色度。JPEG 和影片壓縮常用此格式，因為人眼對亮度敏感、對色度不敏感，可壓縮色度節省空間。也常用於膚色偵測。\n\n【實際應用】顏色過濾追蹤用 HSV、色彩校正用 LAB、皮膚偵測用 YCrCb',
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
            'description': '包含縮放、旋轉、平移、翻轉等操作。旋轉使用 getRotationMatrix2D 建立旋轉矩陣，再用 warpAffine 套用。',
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
            'description': '將灰階影像轉為黑白二值影像。固定閾值需手動設定，Otsu會自動計算最佳閾值，自適應閾值則根據局部區域動態計算。',
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
            'description': '使用鄰域像素的平均值取代中心像素。核心越大，模糊效果越強。這是最基本的卷積濾波操作。',
            'params': [
                {'name': 'ksize', 'label': '核心大小', 'type': 'slider',
                 'min': 3, 'max': 31, 'step': 2, 'default': 5}
            ]
        },
        'blur_gaussian': {
            'name': '高斯濾波',
            'category': '平滑處理',
            'description': '使用高斯函數作為權重，越靠近中心權重越大。比均值濾波更自然，常用於去除高斯雜訊。',
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
            'description': '使用鄰域像素的中位數取代中心像素。對椒鹽雜訊特別有效，能保持邊緣清晰，但運算較慢。',
            'params': [
                {'name': 'ksize', 'label': '核心大小', 'type': 'slider',
                 'min': 3, 'max': 31, 'step': 2, 'default': 5}
            ]
        },
        'blur_bilateral': {
            'name': '雙邊濾波',
            'category': '平滑處理',
            'description': '同時考慮空間距離和顏色相似度的濾波。可以保持邊緣同時平滑區域，常用於美顏磨皮效果。',
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
            'description': '侵蝕(縮小白色區域)、膨脹(擴大白色區域)、開運算(去除小白點)、閉運算(填補小黑洞)、梯度(取得邊緣)。',
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
            'description': 'Sobel 運算子計算影像的一階導數，用於邊緣檢測。可分別計算 X 方向(垂直邊緣)和 Y 方向(水平邊緣)的梯度。',
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
            'description': 'Canny 是最常用的邊緣檢測演算法。步驟：高斯濾波→計算梯度→非極大值抑制→雙閾值檢測→邊緣連接。建議高/低閾值比為 2:1 或 3:1。',
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
            'description': '尋找並繪製影像中的輪廓。可提取面積、周長、外接矩形、最小外接圓、凸包等特徵。常用於物件偵測和計數。',
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
            'description': '直方圖均衡化可增強對比度。CLAHE (對比度受限自適應直方圖均衡化) 可避免過度增強，適合局部對比度調整。',
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
            'description': '將影像從空間域轉換到頻率域。低頻(中心)代表平緩變化，高頻(邊緣)代表快速變化。可用於濾波、壓縮、分析。',
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
            'description': '使用 Haar Cascade 分類器偵測人臉。這是傳統機器學習方法，速度快但精度不如深度學習。可調整縮放比例和鄰居數提高準確度。',
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
