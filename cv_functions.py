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
            'description': '將兩張影像按比例混合：dst = src1 × α + src2 × β + γ。常用於影像融合、浮水印、淡入淡出效果。',
            'params': [
                {'name': 'alpha', 'label': '原圖權重 (α)', 'type': 'slider',
                 'min': 0, 'max': 1, 'step': 0.1, 'default': 0.7}
            ]
        },
        'bitwise': {
            'name': '逐位元邏輯運算',
            'category': '基礎',
            'description': '對影像進行位元運算（AND、OR、XOR、NOT）。常用於遮罩操作、影像合成。AND 運算可用於提取特定區域。',
            'params': [
                {'name': 'operation', 'label': '運算類型', 'type': 'select',
                 'options': [
                     {'value': 'and', 'label': 'AND (交集)'},
                     {'value': 'or', 'label': 'OR (聯集)'},
                     {'value': 'xor', 'label': 'XOR (互斥)'},
                     {'value': 'not', 'label': 'NOT (反轉)'}
                 ], 'default': 'and'},
                {'name': 'shape', 'label': '遮罩形狀', 'type': 'select',
                 'options': [
                     {'value': 'circle', 'label': '圓形'},
                     {'value': 'rectangle', 'label': '矩形'}
                 ], 'default': 'circle'}
            ]
        },
        'mask': {
            'name': '掩模 (Mask)',
            'category': '基礎',
            'description': '使用遮罩選取影像的特定區域。遮罩是一個二值影像，白色區域(255)會被保留，黑色區域(0)會被遮蔽。',
            'params': [
                {'name': 'shape', 'label': '遮罩形狀', 'type': 'select',
                 'options': [
                     {'value': 'circle', 'label': '圓形'},
                     {'value': 'rectangle', 'label': '矩形'},
                     {'value': 'ellipse', 'label': '橢圓形'}
                 ], 'default': 'circle'},
                {'name': 'size', 'label': '遮罩大小 (%)', 'type': 'slider',
                 'min': 10, 'max': 90, 'step': 5, 'default': 50}
            ]
        },
        'bit_plane': {
            'name': '位元平面分解',
            'category': '基礎',
            'description': '8位元灰階影像可分解為8個位元平面。最高位元(MSB)包含最多資訊，最低位元(LSB)包含較多雜訊。LSB常用於影像隱寫術。',
            'params': [
                {'name': 'plane', 'label': '位元平面', 'type': 'slider',
                 'min': 0, 'max': 7, 'step': 1, 'default': 7}
            ]
        },
        'encrypt': {
            'name': '影像加密與解密',
            'category': '基礎',
            'description': '使用 XOR 運算進行簡單加密。XOR 的特性：A⊕B=C，C⊕B=A，相同的金鑰可以加密和解密。',
            'params': [
                {'name': 'seed', 'label': '金鑰種子', 'type': 'slider',
                 'min': 1, 'max': 100, 'step': 1, 'default': 42}
            ]
        },

        # ===== 色彩 =====
        'color_space': {
            'name': '色彩空間轉換',
            'category': '色彩',
            'description': 'BGR是OpenCV預設格式。HSV適合顏色過濾(H:色相,S:飽和度,V:明度)。LAB接近人眼感知。YCrCb用於影片壓縮。',
            'params': [
                {'name': 'space', 'label': '色彩空間', 'type': 'select',
                 'options': [
                     {'value': 'gray', 'label': '灰階 (Grayscale)'},
                     {'value': 'hsv', 'label': 'HSV (色相/飽和度/明度)'},
                     {'value': 'hsv_h', 'label': 'HSV - H通道 (色相)'},
                     {'value': 'hsv_s', 'label': 'HSV - S通道 (飽和度)'},
                     {'value': 'hsv_v', 'label': 'HSV - V通道 (明度)'},
                     {'value': 'lab', 'label': 'LAB'},
                     {'value': 'ycrcb', 'label': 'YCrCb'}
                 ], 'default': 'gray'}
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
        'mask': process_mask,
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
    h, w = img.shape[:2]

    gradient = np.zeros_like(img)
    for i in range(w):
        gradient[:, i] = [int(255 * i / w)] * 3

    result = cv2.addWeighted(img, alpha, gradient, 1-alpha, 0)

    code = f'''import cv2

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# 影像加權和公式: dst = src1 * α + src2 * β + γ
alpha = {alpha}
beta = {1-alpha:.1f}
gamma = 0

result = cv2.addWeighted(img1, alpha, img2, beta, gamma)

# 應用場景:
# - 影像融合
# - 浮水印 (alpha 接近 1)
# - 淡入淡出效果'''
    return result, code


def process_bitwise(img, params):
    operation = params.get('operation', 'and')
    shape = params.get('shape', 'circle')
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    if shape == 'circle':
        cv2.circle(mask, (w//2, h//2), min(h, w)//3, 255, -1)
    else:
        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)

    if operation == 'and':
        result = cv2.bitwise_and(img, img, mask=mask)
    elif operation == 'or':
        mask_3ch = cv2.merge([mask, mask, mask])
        result = cv2.bitwise_or(img, mask_3ch)
    elif operation == 'xor':
        mask_3ch = cv2.merge([mask, mask, mask])
        result = cv2.bitwise_xor(img, mask_3ch)
    else:  # not
        result = cv2.bitwise_not(img)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 建立遮罩 (形狀: {shape})
mask = np.zeros((h, w), dtype=np.uint8)
cv2.{'circle(mask, (w//2, h//2), min(h,w)//3, 255, -1)' if shape == 'circle' else 'rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)'}

# 逐位元 {operation.upper()} 運算
result = cv2.bitwise_{operation}(img, img, mask=mask)

# 其他運算:
# cv2.bitwise_and()  - 交集
# cv2.bitwise_or()   - 聯集
# cv2.bitwise_xor()  - 互斥
# cv2.bitwise_not()  - 反轉'''
    return result, code


def process_mask(img, params):
    shape = params.get('shape', 'circle')
    size = int(params.get('size', 50))
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    radius = int(min(h, w) * size / 100 / 2)

    if shape == 'circle':
        cv2.circle(mask, (w//2, h//2), radius, 255, -1)
    elif shape == 'rectangle':
        x1, y1 = w//2 - radius, h//2 - radius
        x2, y2 = w//2 + radius, h//2 + radius
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    else:  # ellipse
        cv2.ellipse(mask, (w//2, h//2), (radius, radius//2), 0, 0, 360, 255, -1)

    result = cv2.bitwise_and(img, img, mask=mask)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 建立遮罩 (形狀: {shape}, 大小: {size}%)
mask = np.zeros((h, w), dtype=np.uint8)

# 在遮罩上繪製白色區域 (要保留的部分)
# 圓形: cv2.circle(mask, center, radius, 255, -1)
# 矩形: cv2.rectangle(mask, pt1, pt2, 255, -1)
# 橢圓: cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

# 套用遮罩
result = cv2.bitwise_and(img, img, mask=mask)'''
    return result, code


def process_bit_plane(img, params):
    plane = int(params.get('plane', 7))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bit_plane = (gray >> plane) & 1
    result = (bit_plane * 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 提取第 {plane} 個位元平面 (0=LSB, 7=MSB)
bit_plane = (img >> {plane}) & 1
bit_plane = bit_plane * 255  # 放大到 0-255

# 位元平面說明:
# - 平面 7 (MSB): 包含最多視覺資訊
# - 平面 0 (LSB): 主要是雜訊，常用於隱寫術

# 重建影像 (只保留高位元)
reconstructed = (img >> 4) << 4  # 只保留高 4 位元'''
    return result, code


def process_encrypt(img, params):
    seed = int(params.get('seed', 42))
    h, w = img.shape[:2]

    np.random.seed(seed)
    key = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    result = cv2.bitwise_xor(img, key)

    code = f'''import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]

# 產生隨機金鑰 (種子: {seed})
np.random.seed({seed})
key = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

# 加密 (XOR 運算)
encrypted = cv2.bitwise_xor(img, key)

# 解密 (再次 XOR 同樣的金鑰)
np.random.seed({seed})  # 重設種子產生相同金鑰
key = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
decrypted = cv2.bitwise_xor(encrypted, key)'''
    return result, code


def process_color_space(img, params):
    space = params.get('space', 'gray')

    if space == 'gray':
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif space == 'hsv':
        result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    elif space == 'hsv_h':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = cv2.cvtColor(hsv[:,:,0], cv2.COLOR_GRAY2BGR)
    elif space == 'hsv_s':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = cv2.cvtColor(hsv[:,:,1], cv2.COLOR_GRAY2BGR)
    elif space == 'hsv_v':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        result = cv2.cvtColor(hsv[:,:,2], cv2.COLOR_GRAY2BGR)
    elif space == 'lab':
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    elif space == 'ycrcb':
        result = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        result = cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)
    else:
        result = img

    code = f'''import cv2

img = cv2.imread('image.jpg')

# 色彩空間轉換: {space}

# 常用轉換:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# HSV 通道分離
h, s, v = cv2.split(hsv)
# H: 色相 (0-180)
# S: 飽和度 (0-255)
# V: 明度 (0-255)

# 常見應用:
# - HSV: 顏色過濾、物體追蹤
# - LAB: 色彩校正、與人眼感知相近
# - YCrCb: 皮膚偵測、影片壓縮'''
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
