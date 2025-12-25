# -*- coding: utf-8 -*-
"""
NumPy 學習功能模組
提供各種 NumPy 操作的定義和處理函數
"""

import numpy as np

# ===== 效果定義 =====

NUMPY_EFFECTS = {
    # ===== 類別 1: 陣列建立與視覺化 =====
    'array_create': {
        'name': '陣列建立',
        'category': '建立與視覺化',
        'description': '''NumPy 提供多種建立陣列的方式：
• zeros(shape): 建立全 0 陣列
• ones(shape): 建立全 1 陣列
• arange(start, stop, step): 類似 range，建立等差數列
• linspace(start, stop, num): 建立等間距數列
• full(shape, value): 建立指定值的陣列

這些是最常用的陣列建立函數，也是學習 NumPy 的起點。''',
        'params': [
            {'name': 'method', 'label': '建立方式', 'type': 'select',
             'options': [
                 {'value': 'zeros', 'label': 'zeros - 全 0 陣列'},
                 {'value': 'ones', 'label': 'ones - 全 1 陣列'},
                 {'value': 'arange', 'label': 'arange - 等差數列'},
                 {'value': 'linspace', 'label': 'linspace - 等間距數列'},
                 {'value': 'full', 'label': 'full - 指定值填充'}
             ], 'default': 'arange'},
            {'name': 'rows', 'label': '列數', 'type': 'slider', 'min': 1, 'max': 8, 'step': 1, 'default': 4},
            {'name': 'cols', 'label': '欄數', 'type': 'slider', 'min': 1, 'max': 8, 'step': 1, 'default': 6},
            {'name': 'fill_value', 'label': '填充值 (full)', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'default': 7}
        ]
    },

    'array_random': {
        'name': '隨機陣列',
        'category': '建立與視覺化',
        'description': '''NumPy 的隨機數生成：
• random.rand(): 0~1 均勻分布
• random.randint(): 整數隨機
• random.randn(): 標準常態分布 (平均=0, 標準差=1)
• random.normal(): 自訂常態分布
• random.choice(): 從陣列中隨機選取

隨機數在機器學習和統計分析中非常重要。''',
        'params': [
            {'name': 'method', 'label': '隨機方式', 'type': 'select',
             'options': [
                 {'value': 'rand', 'label': 'rand - 0~1 均勻分布'},
                 {'value': 'randint', 'label': 'randint - 整數隨機'},
                 {'value': 'randn', 'label': 'randn - 標準常態分布'},
                 {'value': 'choice', 'label': 'choice - 隨機選取'}
             ], 'default': 'randint'},
            {'name': 'rows', 'label': '列數', 'type': 'slider', 'min': 1, 'max': 8, 'step': 1, 'default': 5},
            {'name': 'cols', 'label': '欄數', 'type': 'slider', 'min': 1, 'max': 8, 'step': 1, 'default': 5},
            {'name': 'low', 'label': '最小值 (randint)', 'type': 'slider', 'min': 0, 'max': 50, 'step': 1, 'default': 0},
            {'name': 'high', 'label': '最大值 (randint)', 'type': 'slider', 'min': 1, 'max': 100, 'step': 1, 'default': 10},
            {'name': 'seed', 'label': '隨機種子', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'default': 42}
        ]
    },

    'array_special': {
        'name': '特殊矩陣',
        'category': '建立與視覺化',
        'description': '''常用的特殊矩陣：
• eye(n): 單位矩陣（對角線為 1）
• diag(v): 對角矩陣（從一維陣列建立）
• tri(n): 下三角矩陣
• triu/tril: 取上/下三角

這些矩陣在線性代數和矩陣運算中經常使用。''',
        'params': [
            {'name': 'method', 'label': '矩陣類型', 'type': 'select',
             'options': [
                 {'value': 'eye', 'label': 'eye - 單位矩陣'},
                 {'value': 'diag', 'label': 'diag - 對角矩陣'},
                 {'value': 'tri', 'label': 'tri - 下三角矩陣'},
                 {'value': 'triu', 'label': 'triu - 取上三角'},
                 {'value': 'tril', 'label': 'tril - 取下三角'}
             ], 'default': 'eye'},
            {'name': 'size', 'label': '矩陣大小', 'type': 'slider', 'min': 2, 'max': 8, 'step': 1, 'default': 5}
        ]
    },

    # ===== 類別 2: 索引與切片 =====
    'indexing': {
        'name': '索引存取',
        'category': '索引與切片',
        'description': '''NumPy 陣列索引：
• arr[i]: 一維索引
• arr[i, j]: 二維索引 (列, 欄)
• arr[i][j]: 連續索引（效率較低）

索引從 0 開始，負數索引表示從尾端算起。
例如 arr[-1] 是最後一個元素。

被選中的元素會以紅色邊框標示。''',
        'params': [
            {'name': 'rows', 'label': '陣列列數', 'type': 'slider', 'min': 3, 'max': 8, 'step': 1, 'default': 5},
            {'name': 'cols', 'label': '陣列欄數', 'type': 'slider', 'min': 3, 'max': 8, 'step': 1, 'default': 6},
            {'name': 'row_idx', 'label': '列索引', 'type': 'slider', 'min': 0, 'max': 7, 'step': 1, 'default': 2},
            {'name': 'col_idx', 'label': '欄索引', 'type': 'slider', 'min': 0, 'max': 7, 'step': 1, 'default': 3}
        ]
    },

    'slicing': {
        'name': '切片操作',
        'category': '索引與切片',
        'description': '''NumPy 切片語法：arr[start:stop:step]
• start: 起始位置（包含）
• stop: 結束位置（不包含）
• step: 步長

範例：
• arr[1:4]: 取索引 1, 2, 3
• arr[::2]: 每隔一個取
• arr[::-1]: 反轉陣列
• arr[1:, :2]: 2D 切片

切片結果會以高亮顯示。''',
        'params': [
            {'name': 'rows', 'label': '陣列列數', 'type': 'slider', 'min': 4, 'max': 8, 'step': 1, 'default': 6},
            {'name': 'cols', 'label': '陣列欄數', 'type': 'slider', 'min': 4, 'max': 8, 'step': 1, 'default': 6},
            {'name': 'row_start', 'label': '列起始', 'type': 'slider', 'min': 0, 'max': 7, 'step': 1, 'default': 1},
            {'name': 'row_stop', 'label': '列結束', 'type': 'slider', 'min': 1, 'max': 8, 'step': 1, 'default': 4},
            {'name': 'col_start', 'label': '欄起始', 'type': 'slider', 'min': 0, 'max': 7, 'step': 1, 'default': 1},
            {'name': 'col_stop', 'label': '欄結束', 'type': 'slider', 'min': 1, 'max': 8, 'step': 1, 'default': 5}
        ]
    },

    'fancy_indexing': {
        'name': '花式索引',
        'category': '索引與切片',
        'description': '''進階索引技巧：
• 布林遮罩：arr[arr > 5] 選取大於 5 的元素
• 整數陣列索引：arr[[0, 2, 4]] 選取特定列
• 條件組合：arr[(arr > 2) & (arr < 8)]

花式索引非常強大，可以實現複雜的資料篩選。
滿足條件的元素會高亮顯示。''',
        'params': [
            {'name': 'rows', 'label': '陣列列數', 'type': 'slider', 'min': 4, 'max': 8, 'step': 1, 'default': 5},
            {'name': 'cols', 'label': '陣列欄數', 'type': 'slider', 'min': 4, 'max': 8, 'step': 1, 'default': 5},
            {'name': 'condition', 'label': '篩選條件', 'type': 'select',
             'options': [
                 {'value': 'gt', 'label': '大於閾值'},
                 {'value': 'lt', 'label': '小於閾值'},
                 {'value': 'eq', 'label': '等於閾值'},
                 {'value': 'even', 'label': '偶數'},
                 {'value': 'odd', 'label': '奇數'},
                 {'value': 'range', 'label': '在範圍內'}
             ], 'default': 'gt'},
            {'name': 'threshold', 'label': '閾值', 'type': 'slider', 'min': 0, 'max': 30, 'step': 1, 'default': 15},
            {'name': 'seed', 'label': '隨機種子', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'default': 42}
        ]
    },

    # ===== 類別 3: 形狀操作 =====
    'reshape': {
        'name': '改變形狀',
        'category': '形狀操作',
        'description': '''reshape() 改變陣列形狀：
• 元素總數必須相同
• -1 表示自動計算該維度
• 資料順序（row-major）保持不變

範例：
• arr.reshape(2, 6): 變成 2 列 6 欄
• arr.reshape(-1): 攤平成一維
• arr.reshape(3, -1): 3 列，欄數自動算

reshape 不會改變資料本身，只改變「看」資料的方式。''',
        'params': [
            {'name': 'original_rows', 'label': '原始列數', 'type': 'slider', 'min': 2, 'max': 6, 'step': 1, 'default': 3},
            {'name': 'original_cols', 'label': '原始欄數', 'type': 'slider', 'min': 2, 'max': 8, 'step': 1, 'default': 4},
            {'name': 'new_rows', 'label': '新列數', 'type': 'slider', 'min': 1, 'max': 12, 'step': 1, 'default': 4},
            {'name': 'new_cols', 'label': '新欄數', 'type': 'slider', 'min': 1, 'max': 12, 'step': 1, 'default': 3}
        ]
    },

    'transpose': {
        'name': '轉置',
        'category': '形狀操作',
        'description': '''轉置操作：
• arr.T: 最簡單的轉置
• arr.transpose(): 同上
• np.swapaxes(arr, 0, 1): 交換軸

轉置就是行列互換：原本的第 i 列第 j 欄 → 變成第 j 列第 i 欄

對於 2D 陣列，就是沿對角線翻轉。''',
        'params': [
            {'name': 'rows', 'label': '列數', 'type': 'slider', 'min': 2, 'max': 6, 'step': 1, 'default': 3},
            {'name': 'cols', 'label': '欄數', 'type': 'slider', 'min': 2, 'max': 8, 'step': 1, 'default': 5}
        ]
    },

    'stack_split': {
        'name': '堆疊與拆分',
        'category': '形狀操作',
        'description': '''陣列的組合與拆分：
堆疊：
• np.vstack(): 垂直堆疊（上下）
• np.hstack(): 水平堆疊（左右）
• np.concatenate(): 通用合併

拆分：
• np.vsplit(): 垂直拆分
• np.hsplit(): 水平拆分
• np.split(): 通用拆分''',
        'params': [
            {'name': 'operation', 'label': '操作類型', 'type': 'select',
             'options': [
                 {'value': 'vstack', 'label': 'vstack - 垂直堆疊'},
                 {'value': 'hstack', 'label': 'hstack - 水平堆疊'},
                 {'value': 'vsplit', 'label': 'vsplit - 垂直拆分'},
                 {'value': 'hsplit', 'label': 'hsplit - 水平拆分'}
             ], 'default': 'vstack'},
            {'name': 'rows', 'label': '陣列列數', 'type': 'slider', 'min': 2, 'max': 6, 'step': 1, 'default': 4},
            {'name': 'cols', 'label': '陣列欄數', 'type': 'slider', 'min': 2, 'max': 6, 'step': 1, 'default': 4}
        ]
    },

    # ===== 類別 4: 數學運算 =====
    'element_wise': {
        'name': '元素運算',
        'category': '數學運算',
        'description': '''元素對元素（element-wise）運算：
• +, -, *, /: 四則運算
• **: 次方
• np.sqrt(): 開根號
• np.exp(): 指數
• np.log(): 對數

兩個形狀相同的陣列進行運算時，對應位置的元素會一一運算。''',
        'params': [
            {'name': 'operation', 'label': '運算類型', 'type': 'select',
             'options': [
                 {'value': 'add', 'label': '加法 (+)'},
                 {'value': 'sub', 'label': '減法 (-)'},
                 {'value': 'mul', 'label': '乘法 (*)'},
                 {'value': 'div', 'label': '除法 (/)'},
                 {'value': 'pow', 'label': '次方 (**)'},
                 {'value': 'mod', 'label': '餘數 (%)'}
             ], 'default': 'add'},
            {'name': 'rows', 'label': '陣列大小', 'type': 'slider', 'min': 2, 'max': 5, 'step': 1, 'default': 3},
            {'name': 'seed', 'label': '隨機種子', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'default': 42}
        ]
    },

    'broadcasting': {
        'name': '廣播機制',
        'category': '數學運算',
        'description': '''廣播（Broadcasting）讓不同形狀的陣列可以運算：

規則：
1. 維度不足的陣列在前面補 1
2. 大小為 1 的維度會擴展以匹配

範例：
• (3, 4) + (4,) → (3, 4) + (1, 4) → 逐列相加
• (3, 1) + (1, 4) → (3, 4)

廣播讓程式碼更簡潔，也更有效率！''',
        'params': [
            {'name': 'broadcast_type', 'label': '廣播類型', 'type': 'select',
             'options': [
                 {'value': 'scalar', 'label': '純量 + 陣列'},
                 {'value': 'row', 'label': '列向量 + 矩陣'},
                 {'value': 'col', 'label': '欄向量 + 矩陣'},
                 {'value': 'outer', 'label': '列 × 欄（外積）'}
             ], 'default': 'row'},
            {'name': 'rows', 'label': '矩陣列數', 'type': 'slider', 'min': 2, 'max': 5, 'step': 1, 'default': 4},
            {'name': 'cols', 'label': '矩陣欄數', 'type': 'slider', 'min': 2, 'max': 5, 'step': 1, 'default': 4}
        ]
    },

    'dot_product': {
        'name': '矩陣乘法',
        'category': '數學運算',
        'description': '''矩陣乘法 vs 元素乘法：
• A @ B 或 np.dot(A, B): 矩陣乘法
• A * B: 元素對應相乘

矩陣乘法規則：
(m, n) @ (n, p) → (m, p)
第一個矩陣的欄數必須等於第二個矩陣的列數。

計算方式：C[i,j] = Σ A[i,k] × B[k,j]''',
        'params': [
            {'name': 'product_type', 'label': '乘法類型', 'type': 'select',
             'options': [
                 {'value': 'dot', 'label': '矩陣乘法 (@)'},
                 {'value': 'element', 'label': '元素乘法 (*)'}
             ], 'default': 'dot'},
            {'name': 'rows_a', 'label': 'A 的列數', 'type': 'slider', 'min': 2, 'max': 5, 'step': 1, 'default': 3},
            {'name': 'cols_a', 'label': 'A 的欄數', 'type': 'slider', 'min': 2, 'max': 5, 'step': 1, 'default': 4},
            {'name': 'cols_b', 'label': 'B 的欄數', 'type': 'slider', 'min': 2, 'max': 5, 'step': 1, 'default': 2},
            {'name': 'seed', 'label': '隨機種子', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'default': 42}
        ]
    },

    # ===== 類別 5: 統計與分析 =====
    'statistics': {
        'name': '統計函數',
        'category': '統計與分析',
        'description': '''常用統計函數：
• np.sum(): 總和
• np.mean(): 平均值
• np.std(): 標準差
• np.var(): 變異數
• np.min() / np.max(): 最小 / 最大值
• np.argmin() / np.argmax(): 最小 / 最大值的索引

可以指定 axis 參數：
• axis=0: 沿列（垂直）計算
• axis=1: 沿欄（水平）計算
• 不指定: 對整個陣列計算''',
        'params': [
            {'name': 'function', 'label': '統計函數', 'type': 'select',
             'options': [
                 {'value': 'sum', 'label': 'sum - 總和'},
                 {'value': 'mean', 'label': 'mean - 平均值'},
                 {'value': 'std', 'label': 'std - 標準差'},
                 {'value': 'min', 'label': 'min - 最小值'},
                 {'value': 'max', 'label': 'max - 最大值'},
                 {'value': 'cumsum', 'label': 'cumsum - 累計和'}
             ], 'default': 'mean'},
            {'name': 'axis', 'label': '計算軸向', 'type': 'select',
             'options': [
                 {'value': 'none', 'label': '全部 (無 axis)'},
                 {'value': '0', 'label': 'axis=0 (沿列)'},
                 {'value': '1', 'label': 'axis=1 (沿欄)'}
             ], 'default': 'none'},
            {'name': 'rows', 'label': '列數', 'type': 'slider', 'min': 2, 'max': 6, 'step': 1, 'default': 4},
            {'name': 'cols', 'label': '欄數', 'type': 'slider', 'min': 2, 'max': 6, 'step': 1, 'default': 5},
            {'name': 'seed', 'label': '隨機種子', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'default': 42}
        ]
    },

    'sort_search': {
        'name': '排序與搜尋',
        'category': '統計與分析',
        'description': '''排序與搜尋函數：
排序：
• np.sort(arr): 排序後的陣列
• np.argsort(arr): 排序後的索引
• arr.sort(): 原地排序（會改變原陣列）

搜尋：
• np.where(condition): 找出符合條件的索引
• np.searchsorted(arr, v): 二分搜尋插入位置
• np.unique(arr): 找出唯一值''',
        'params': [
            {'name': 'operation', 'label': '操作類型', 'type': 'select',
             'options': [
                 {'value': 'sort', 'label': 'sort - 排序'},
                 {'value': 'argsort', 'label': 'argsort - 排序索引'},
                 {'value': 'where', 'label': 'where - 條件搜尋'},
                 {'value': 'unique', 'label': 'unique - 唯一值'}
             ], 'default': 'sort'},
            {'name': 'axis', 'label': '操作軸向', 'type': 'select',
             'options': [
                 {'value': 'none', 'label': '攤平排序'},
                 {'value': '0', 'label': 'axis=0 (每欄排序)'},
                 {'value': '1', 'label': 'axis=1 (每列排序)'}
             ], 'default': 'none'},
            {'name': 'rows', 'label': '列數', 'type': 'slider', 'min': 2, 'max': 6, 'step': 1, 'default': 4},
            {'name': 'cols', 'label': '欄數', 'type': 'slider', 'min': 2, 'max': 6, 'step': 1, 'default': 5},
            {'name': 'threshold', 'label': '閾值 (where)', 'type': 'slider', 'min': 0, 'max': 50, 'step': 1, 'default': 15},
            {'name': 'seed', 'label': '隨機種子', 'type': 'slider', 'min': 0, 'max': 100, 'step': 1, 'default': 42}
        ]
    },

    # ===== 類別 6: 圖片即陣列 =====
    'image_as_array': {
        'name': '圖片即陣列',
        'category': '圖片即陣列',
        'description': '''在 NumPy 和 OpenCV 中，圖片就是陣列：
• 灰階圖: (height, width) - 2D 陣列
• 彩色圖: (height, width, 3) - 3D 陣列
• 每個元素是 0~255 的像素值

OpenCV 使用 BGR 順序（不是 RGB）！
• img.shape: 取得形狀
• img.dtype: 資料型態（通常是 uint8）
• img.size: 總元素數''',
        'params': [],
        'requires_image': True
    },

    'channel_slice': {
        'name': '通道切片',
        'category': '圖片即陣列',
        'description': '''用陣列切片取出圖片通道：
• img[:, :, 0]: B 通道（藍色）
• img[:, :, 1]: G 通道（綠色）
• img[:, :, 2]: R 通道（紅色）

技巧：
• 將某通道設為 0 可以移除該顏色
• 只保留單一通道可以分析該顏色分布
• 交換通道可以改變顏色效果''',
        'params': [
            {'name': 'channel', 'label': '選取通道', 'type': 'select',
             'options': [
                 {'value': 'b', 'label': 'B 通道 (藍色)'},
                 {'value': 'g', 'label': 'G 通道 (綠色)'},
                 {'value': 'r', 'label': 'R 通道 (紅色)'},
                 {'value': 'no_b', 'label': '移除 B（黃色調）'},
                 {'value': 'no_g', 'label': '移除 G（紫色調）'},
                 {'value': 'no_r', 'label': '移除 R（青色調）'}
             ], 'default': 'b'},
            {'name': 'show_gray', 'label': '顯示為灰階', 'type': 'checkbox', 'default': False}
        ],
        'requires_image': True
    },

    'roi_crop': {
        'name': '區域裁切',
        'category': '圖片即陣列',
        'description': '''使用切片語法裁切圖片區域（ROI）：
roi = img[y1:y2, x1:x2]

注意座標順序：
• 第一個維度是 Y（列）
• 第二個維度是 X（欄）

這是因為陣列的索引順序是 [row, col]，
對應到圖片就是 [y, x]。''',
        'params': [
            {'name': 'x_start', 'label': 'X 起點 (%)', 'type': 'slider', 'min': 0, 'max': 80, 'step': 5, 'default': 20},
            {'name': 'x_end', 'label': 'X 終點 (%)', 'type': 'slider', 'min': 20, 'max': 100, 'step': 5, 'default': 80},
            {'name': 'y_start', 'label': 'Y 起點 (%)', 'type': 'slider', 'min': 0, 'max': 80, 'step': 5, 'default': 20},
            {'name': 'y_end', 'label': 'Y 終點 (%)', 'type': 'slider', 'min': 20, 'max': 100, 'step': 5, 'default': 80}
        ],
        'requires_image': True
    },

    'pixel_math': {
        'name': '像素運算',
        'category': '圖片即陣列',
        'description': '''對像素值進行數學運算：
• 加減常數: 調整亮度
• 乘除常數: 調整對比
• 裁切到 0~255: np.clip(img, 0, 255)

注意溢位問題：
uint8 的範圍是 0~255，超過會溢位！
解決方法：先轉成 int/float，運算後再轉回 uint8。''',
        'params': [
            {'name': 'operation', 'label': '運算類型', 'type': 'select',
             'options': [
                 {'value': 'add', 'label': '加法（亮度+）'},
                 {'value': 'sub', 'label': '減法（亮度-）'},
                 {'value': 'mul', 'label': '乘法（對比↑）'},
                 {'value': 'div', 'label': '除法（對比↓）'},
                 {'value': 'invert', 'label': '反轉（負片）'},
                 {'value': 'power', 'label': 'Gamma 校正'}
             ], 'default': 'add'},
            {'name': 'value', 'label': '運算值', 'type': 'slider', 'min': -100, 'max': 100, 'step': 5, 'default': 50},
            {'name': 'gamma', 'label': 'Gamma 值', 'type': 'slider', 'min': 0.1, 'max': 3, 'step': 0.1, 'default': 1.0}
        ],
        'requires_image': True
    }
}


def get_all_numpy_effects():
    """取得所有 NumPy 效果的定義"""
    return NUMPY_EFFECTS


def generate_sample_array(array_type, shape):
    """產生範例陣列"""
    rows, cols = shape

    if array_type == 'arange':
        return np.arange(rows * cols).reshape(rows, cols)
    elif array_type == 'random':
        return np.random.randint(0, 100, (rows, cols))
    elif array_type == 'ones':
        return np.ones((rows, cols), dtype=int)
    elif array_type == 'zeros':
        return np.zeros((rows, cols), dtype=int)
    else:
        return np.arange(rows * cols).reshape(rows, cols)


# ===== 處理函數 =====

def process_numpy_operation(effect, params, img=None):
    """
    處理 NumPy 操作
    返回: (input_data, output_data, code, extra_info)
    - input_data: 輸入陣列或陣列列表
    - output_data: 輸出陣列或數值
    - code: Python 程式碼
    - extra_info: 額外視覺化資訊（高亮位置等）
    """
    processor = processors.get(effect)
    if processor:
        return processor(params, img)
    else:
        arr = np.arange(12).reshape(3, 4)
        return arr, arr, "# 未知操作", {}


# ===== 各效果的處理函數 =====

def process_array_create(params, img=None):
    method = params.get('method', 'arange')
    rows = int(params.get('rows', 4))
    cols = int(params.get('cols', 6))
    fill_value = int(params.get('fill_value', 7))

    if method == 'zeros':
        arr = np.zeros((rows, cols), dtype=int)
        code = f"import numpy as np\n\n# 建立全 0 陣列\narr = np.zeros(({rows}, {cols}), dtype=int)\nprint(arr)"
    elif method == 'ones':
        arr = np.ones((rows, cols), dtype=int)
        code = f"import numpy as np\n\n# 建立全 1 陣列\narr = np.ones(({rows}, {cols}), dtype=int)\nprint(arr)"
    elif method == 'arange':
        arr = np.arange(rows * cols).reshape(rows, cols)
        code = f"import numpy as np\n\n# 建立等差數列並 reshape\narr = np.arange({rows * cols}).reshape({rows}, {cols})\nprint(arr)"
    elif method == 'linspace':
        arr = np.linspace(0, 1, rows * cols).reshape(rows, cols)
        arr = np.round(arr, 2)
        code = f"import numpy as np\n\n# 建立等間距數列\narr = np.linspace(0, 1, {rows * cols}).reshape({rows}, {cols})\nprint(arr)"
    elif method == 'full':
        arr = np.full((rows, cols), fill_value, dtype=int)
        code = f"import numpy as np\n\n# 建立指定值填充的陣列\narr = np.full(({rows}, {cols}), {fill_value}, dtype=int)\nprint(arr)"
    else:
        arr = np.arange(rows * cols).reshape(rows, cols)
        code = f"arr = np.arange({rows * cols}).reshape({rows}, {cols})"

    return None, arr, code, {}


def process_array_random(params, img=None):
    method = params.get('method', 'randint')
    rows = int(params.get('rows', 5))
    cols = int(params.get('cols', 5))
    low = int(params.get('low', 0))
    high = int(params.get('high', 10))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)

    if method == 'rand':
        arr = np.round(np.random.rand(rows, cols), 2)
        code = f"import numpy as np\n\nnp.random.seed({seed})\n\n# 0~1 均勻分布\narr = np.random.rand({rows}, {cols})\nprint(arr)"
    elif method == 'randint':
        arr = np.random.randint(low, high + 1, (rows, cols))
        code = f"import numpy as np\n\nnp.random.seed({seed})\n\n# 整數隨機 [{low}, {high}]\narr = np.random.randint({low}, {high + 1}, ({rows}, {cols}))\nprint(arr)"
    elif method == 'randn':
        arr = np.round(np.random.randn(rows, cols), 2)
        code = f"import numpy as np\n\nnp.random.seed({seed})\n\n# 標準常態分布\narr = np.random.randn({rows}, {cols})\nprint(arr)"
    elif method == 'choice':
        choices = [0, 1, 2, 5, 10]
        arr = np.random.choice(choices, (rows, cols))
        code = f"import numpy as np\n\nnp.random.seed({seed})\n\n# 從指定陣列隨機選取\nchoices = {choices}\narr = np.random.choice(choices, ({rows}, {cols}))\nprint(arr)"
    else:
        arr = np.random.randint(0, 10, (rows, cols))
        code = f"arr = np.random.randint(0, 10, ({rows}, {cols}))"

    return None, arr, code, {}


def process_array_special(params, img=None):
    method = params.get('method', 'eye')
    size = int(params.get('size', 5))

    if method == 'eye':
        arr = np.eye(size, dtype=int)
        code = f"import numpy as np\n\n# 單位矩陣\narr = np.eye({size}, dtype=int)\nprint(arr)"
    elif method == 'diag':
        diag_values = list(range(1, size + 1))
        arr = np.diag(diag_values)
        code = f"import numpy as np\n\n# 對角矩陣\nvalues = {diag_values}\narr = np.diag(values)\nprint(arr)"
    elif method == 'tri':
        arr = np.tri(size, dtype=int)
        code = f"import numpy as np\n\n# 下三角矩陣\narr = np.tri({size}, dtype=int)\nprint(arr)"
    elif method == 'triu':
        base = np.arange(size * size).reshape(size, size) + 1
        arr = np.triu(base)
        code = f"import numpy as np\n\n# 取上三角\nbase = np.arange({size * size}).reshape({size}, {size}) + 1\narr = np.triu(base)\nprint(arr)"
    elif method == 'tril':
        base = np.arange(size * size).reshape(size, size) + 1
        arr = np.tril(base)
        code = f"import numpy as np\n\n# 取下三角\nbase = np.arange({size * size}).reshape({size}, {size}) + 1\narr = np.tril(base)\nprint(arr)"
    else:
        arr = np.eye(size, dtype=int)
        code = f"arr = np.eye({size})"

    return None, arr, code, {}


def process_indexing(params, img=None):
    rows = int(params.get('rows', 5))
    cols = int(params.get('cols', 6))
    row_idx = int(params.get('row_idx', 2))
    col_idx = int(params.get('col_idx', 3))

    # 確保索引在範圍內
    row_idx = min(row_idx, rows - 1)
    col_idx = min(col_idx, cols - 1)

    arr = np.arange(rows * cols).reshape(rows, cols)
    value = arr[row_idx, col_idx]

    code = f"""import numpy as np

# 建立陣列
arr = np.arange({rows * cols}).reshape({rows}, {cols})
print("陣列:")
print(arr)

# 索引存取
value = arr[{row_idx}, {col_idx}]
print(f"\\narr[{row_idx}, {col_idx}] = {{value}}")"""

    extra_info = {
        'highlight': [(row_idx, col_idx)],
        'selected_value': int(value)
    }

    return arr, value, code, extra_info


def process_slicing(params, img=None):
    rows = int(params.get('rows', 6))
    cols = int(params.get('cols', 6))
    row_start = int(params.get('row_start', 1))
    row_stop = int(params.get('row_stop', 4))
    col_start = int(params.get('col_start', 1))
    col_stop = int(params.get('col_stop', 5))

    # 確保範圍有效
    row_start = min(row_start, rows - 1)
    row_stop = min(row_stop, rows)
    col_start = min(col_start, cols - 1)
    col_stop = min(col_stop, cols)

    if row_start >= row_stop:
        row_stop = row_start + 1
    if col_start >= col_stop:
        col_stop = col_start + 1

    arr = np.arange(rows * cols).reshape(rows, cols)
    sliced = arr[row_start:row_stop, col_start:col_stop]

    code = f"""import numpy as np

# 建立陣列
arr = np.arange({rows * cols}).reshape({rows}, {cols})
print("原始陣列:")
print(arr)

# 切片操作
sliced = arr[{row_start}:{row_stop}, {col_start}:{col_stop}]
print(f"\\narr[{row_start}:{row_stop}, {col_start}:{col_stop}] =")
print(sliced)"""

    # 高亮切片範圍
    highlights = []
    for r in range(row_start, row_stop):
        for c in range(col_start, col_stop):
            highlights.append((r, c))

    extra_info = {
        'highlight': highlights,
        'slice_bounds': {
            'row_start': row_start, 'row_stop': row_stop,
            'col_start': col_start, 'col_stop': col_stop
        }
    }

    return arr, sliced, code, extra_info


def process_fancy_indexing(params, img=None):
    rows = int(params.get('rows', 5))
    cols = int(params.get('cols', 5))
    condition = params.get('condition', 'gt')
    threshold = int(params.get('threshold', 15))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)
    arr = np.random.randint(0, 30, (rows, cols))

    if condition == 'gt':
        mask = arr > threshold
        condition_str = f"arr > {threshold}"
    elif condition == 'lt':
        mask = arr < threshold
        condition_str = f"arr < {threshold}"
    elif condition == 'eq':
        mask = arr == threshold
        condition_str = f"arr == {threshold}"
    elif condition == 'even':
        mask = arr % 2 == 0
        condition_str = "arr % 2 == 0"
    elif condition == 'odd':
        mask = arr % 2 == 1
        condition_str = "arr % 2 == 1"
    elif condition == 'range':
        low = max(0, threshold - 5)
        high = threshold + 5
        mask = (arr >= low) & (arr <= high)
        condition_str = f"(arr >= {low}) & (arr <= {high})"
    else:
        mask = arr > threshold
        condition_str = f"arr > {threshold}"

    selected = arr[mask]

    code = f"""import numpy as np

np.random.seed({seed})
arr = np.random.randint(0, 30, ({rows}, {cols}))
print("原始陣列:")
print(arr)

# 布林遮罩
mask = {condition_str}
print("\\n布林遮罩:")
print(mask)

# 選取符合條件的元素
selected = arr[mask]
print(f"\\n符合條件的元素: {{selected}}")"""

    # 高亮符合條件的位置
    highlights = []
    for r in range(rows):
        for c in range(cols):
            if mask[r, c]:
                highlights.append((r, c))

    extra_info = {
        'highlight': highlights,
        'mask': mask.tolist(),
        'condition': condition_str
    }

    return arr, selected, code, extra_info


def process_reshape(params, img=None):
    orig_rows = int(params.get('original_rows', 3))
    orig_cols = int(params.get('original_cols', 4))
    new_rows = int(params.get('new_rows', 4))
    new_cols = int(params.get('new_cols', 3))

    total = orig_rows * orig_cols

    # 檢查是否可以 reshape
    if new_rows * new_cols != total:
        # 嘗試調整
        new_cols = total // new_rows
        if new_rows * new_cols != total:
            new_rows = 1
            new_cols = total

    arr = np.arange(total).reshape(orig_rows, orig_cols)
    reshaped = arr.reshape(new_rows, new_cols)

    code = f"""import numpy as np

# 原始陣列
arr = np.arange({total}).reshape({orig_rows}, {orig_cols})
print("原始陣列 (shape: {arr.shape}):")
print(arr)

# reshape 改變形狀
reshaped = arr.reshape({new_rows}, {new_cols})
print(f"\\nreshape 後 (shape: {{reshaped.shape}}):")
print(reshaped)

# 注意：元素總數必須相同
# {orig_rows} × {orig_cols} = {new_rows} × {new_cols} = {total}"""

    extra_info = {
        'original_shape': (orig_rows, orig_cols),
        'new_shape': (new_rows, new_cols)
    }

    return arr, reshaped, code, extra_info


def process_transpose(params, img=None):
    rows = int(params.get('rows', 3))
    cols = int(params.get('cols', 5))

    arr = np.arange(rows * cols).reshape(rows, cols)
    transposed = arr.T

    code = f"""import numpy as np

# 原始陣列
arr = np.arange({rows * cols}).reshape({rows}, {cols})
print("原始陣列 (shape: {arr.shape}):")
print(arr)

# 轉置
transposed = arr.T  # 或 arr.transpose()
print(f"\\n轉置後 (shape: {{transposed.shape}}):")
print(transposed)

# 轉置後 arr[i, j] → transposed[j, i]"""

    extra_info = {
        'original_shape': arr.shape,
        'transposed_shape': transposed.shape
    }

    return arr, transposed, code, extra_info


def process_stack_split(params, img=None):
    operation = params.get('operation', 'vstack')
    rows = int(params.get('rows', 4))
    cols = int(params.get('cols', 4))

    if operation in ['vstack', 'hstack']:
        arr1 = np.arange(rows * cols).reshape(rows, cols)
        arr2 = np.arange(rows * cols, rows * cols * 2).reshape(rows, cols)

        if operation == 'vstack':
            result = np.vstack([arr1, arr2])
            code = f"""import numpy as np

# 建立兩個陣列
arr1 = np.arange({rows * cols}).reshape({rows}, {cols})
arr2 = np.arange({rows * cols}, {rows * cols * 2}).reshape({rows}, {cols})

print("arr1:")
print(arr1)
print("\\narr2:")
print(arr2)

# 垂直堆疊（上下合併）
result = np.vstack([arr1, arr2])
print("\\nvstack 結果:")
print(result)"""
        else:  # hstack
            result = np.hstack([arr1, arr2])
            code = f"""import numpy as np

# 建立兩個陣列
arr1 = np.arange({rows * cols}).reshape({rows}, {cols})
arr2 = np.arange({rows * cols}, {rows * cols * 2}).reshape({rows}, {cols})

print("arr1:")
print(arr1)
print("\\narr2:")
print(arr2)

# 水平堆疊（左右合併）
result = np.hstack([arr1, arr2])
print("\\nhstack 結果:")
print(result)"""

        extra_info = {
            'input_arrays': [arr1.tolist(), arr2.tolist()],
            'is_stack': True
        }
        return [arr1, arr2], result, code, extra_info

    else:  # vsplit, hsplit
        arr = np.arange(rows * cols).reshape(rows, cols)

        if operation == 'vsplit':
            n_splits = 2 if rows >= 2 else 1
            if rows % n_splits != 0:
                n_splits = 1
            result = np.vsplit(arr, n_splits)
            code = f"""import numpy as np

arr = np.arange({rows * cols}).reshape({rows}, {cols})
print("原始陣列:")
print(arr)

# 垂直拆分成 {n_splits} 份
result = np.vsplit(arr, {n_splits})
for i, part in enumerate(result):
    print(f"\\n第 {{i+1}} 部分:")
    print(part)"""
        else:  # hsplit
            n_splits = 2 if cols >= 2 else 1
            if cols % n_splits != 0:
                n_splits = 1
            result = np.hsplit(arr, n_splits)
            code = f"""import numpy as np

arr = np.arange({rows * cols}).reshape({rows}, {cols})
print("原始陣列:")
print(arr)

# 水平拆分成 {n_splits} 份
result = np.hsplit(arr, {n_splits})
for i, part in enumerate(result):
    print(f"\\n第 {{i+1}} 部分:")
    print(part)"""

        extra_info = {
            'is_split': True,
            'n_splits': len(result)
        }
        return arr, [r.tolist() for r in result], code, extra_info


def process_element_wise(params, img=None):
    operation = params.get('operation', 'add')
    size = int(params.get('rows', 3))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)
    arr1 = np.random.randint(1, 10, (size, size))
    arr2 = np.random.randint(1, 10, (size, size))

    op_map = {
        'add': ('+', np.add),
        'sub': ('-', np.subtract),
        'mul': ('*', np.multiply),
        'div': ('/', np.divide),
        'pow': ('**', np.power),
        'mod': ('%', np.mod)
    }

    op_sym, op_func = op_map.get(operation, ('+', np.add))

    if operation == 'div':
        result = np.round(op_func(arr1.astype(float), arr2), 2)
    else:
        result = op_func(arr1, arr2)

    code = f"""import numpy as np

np.random.seed({seed})
arr1 = np.random.randint(1, 10, ({size}, {size}))
arr2 = np.random.randint(1, 10, ({size}, {size}))

print("arr1:")
print(arr1)
print("\\narr2:")
print(arr2)

# 元素對應運算
result = arr1 {op_sym} arr2
print(f"\\narr1 {op_sym} arr2 =")
print(result)"""

    extra_info = {
        'operation_symbol': op_sym,
        'input_arrays': [arr1.tolist(), arr2.tolist()]
    }

    return [arr1, arr2], result, code, extra_info


def process_broadcasting(params, img=None):
    broadcast_type = params.get('broadcast_type', 'row')
    rows = int(params.get('rows', 4))
    cols = int(params.get('cols', 4))

    arr = np.arange(rows * cols).reshape(rows, cols)

    if broadcast_type == 'scalar':
        scalar = 10
        result = arr + scalar
        code = f"""import numpy as np

arr = np.arange({rows * cols}).reshape({rows}, {cols})
print("陣列:")
print(arr)

# 純量 + 陣列（廣播）
result = arr + {scalar}
print(f"\\narr + {scalar} =")
print(result)

# 純量會被「廣播」成與陣列相同形狀"""
        extra_info = {'scalar': scalar, 'broadcast_type': 'scalar'}
        return arr, result, code, extra_info

    elif broadcast_type == 'row':
        row_vec = np.arange(cols) * 10
        result = arr + row_vec
        code = f"""import numpy as np

arr = np.arange({rows * cols}).reshape({rows}, {cols})
row_vec = np.arange({cols}) * 10  # shape: ({cols},)

print("陣列 (shape: {arr.shape}):")
print(arr)
print(f"\\n列向量 (shape: {{row_vec.shape}}):")
print(row_vec)

# 列向量 + 矩陣（沿列廣播）
result = arr + row_vec
print("\\narr + row_vec =")
print(result)

# row_vec 被廣播成 ({rows}, {cols})"""
        extra_info = {'vector': row_vec.tolist(), 'broadcast_type': 'row'}
        return [arr, row_vec], result, code, extra_info

    elif broadcast_type == 'col':
        col_vec = (np.arange(rows) * 10).reshape(-1, 1)
        result = arr + col_vec
        code = f"""import numpy as np

arr = np.arange({rows * cols}).reshape({rows}, {cols})
col_vec = (np.arange({rows}) * 10).reshape(-1, 1)  # shape: ({rows}, 1)

print("陣列 (shape: {arr.shape}):")
print(arr)
print(f"\\n欄向量 (shape: {{col_vec.shape}}):")
print(col_vec)

# 欄向量 + 矩陣（沿欄廣播）
result = arr + col_vec
print("\\narr + col_vec =")
print(result)

# col_vec 被廣播成 ({rows}, {cols})"""
        extra_info = {'vector': col_vec.tolist(), 'broadcast_type': 'col'}
        return [arr, col_vec], result, code, extra_info

    else:  # outer
        row_vec = np.arange(1, cols + 1)
        col_vec = np.arange(1, rows + 1).reshape(-1, 1)
        result = row_vec * col_vec
        code = f"""import numpy as np

row_vec = np.arange(1, {cols + 1})  # shape: ({cols},)
col_vec = np.arange(1, {rows + 1}).reshape(-1, 1)  # shape: ({rows}, 1)

print("列向量:")
print(row_vec)
print("\\n欄向量:")
print(col_vec)

# 外積（乘法表！）
result = row_vec * col_vec
print("\\nrow_vec * col_vec =")
print(result)

# 這就是乘法表的原理！"""
        extra_info = {'row_vec': row_vec.tolist(), 'col_vec': col_vec.flatten().tolist(), 'broadcast_type': 'outer'}
        return [row_vec, col_vec], result, code, extra_info


def process_dot_product(params, img=None):
    product_type = params.get('product_type', 'dot')
    rows_a = int(params.get('rows_a', 3))
    cols_a = int(params.get('cols_a', 4))
    cols_b = int(params.get('cols_b', 2))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)
    A = np.random.randint(0, 5, (rows_a, cols_a))

    if product_type == 'dot':
        B = np.random.randint(0, 5, (cols_a, cols_b))
        result = A @ B
        code = f"""import numpy as np

np.random.seed({seed})
A = np.random.randint(0, 5, ({rows_a}, {cols_a}))
B = np.random.randint(0, 5, ({cols_a}, {cols_b}))

print("A (shape: {A.shape}):")
print(A)
print("\\nB (shape: {B.shape}):")
print(B)

# 矩陣乘法
result = A @ B  # 或 np.dot(A, B)
print(f"\\nA @ B (shape: {{result.shape}}):")
print(result)

# ({rows_a}, {cols_a}) @ ({cols_a}, {cols_b}) = ({rows_a}, {cols_b})"""
    else:  # element
        B = np.random.randint(0, 5, (rows_a, cols_a))
        result = A * B
        code = f"""import numpy as np

np.random.seed({seed})
A = np.random.randint(0, 5, ({rows_a}, {cols_a}))
B = np.random.randint(0, 5, ({rows_a}, {cols_a}))

print("A:")
print(A)
print("\\nB:")
print(B)

# 元素對應相乘（不是矩陣乘法！）
result = A * B
print("\\nA * B =")
print(result)"""

    extra_info = {
        'product_type': product_type,
        'A_shape': A.shape,
        'B_shape': B.shape
    }

    return [A, B], result, code, extra_info


def process_statistics(params, img=None):
    function = params.get('function', 'mean')
    axis_str = params.get('axis', 'none')
    rows = int(params.get('rows', 4))
    cols = int(params.get('cols', 5))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)
    arr = np.random.randint(0, 100, (rows, cols))

    axis = None if axis_str == 'none' else int(axis_str)
    axis_desc = '全部' if axis is None else f'axis={axis}'

    func_map = {
        'sum': (np.sum, 'sum'),
        'mean': (np.mean, 'mean'),
        'std': (np.std, 'std'),
        'min': (np.min, 'min'),
        'max': (np.max, 'max'),
        'cumsum': (np.cumsum, 'cumsum')
    }

    func, func_name = func_map.get(function, (np.mean, 'mean'))

    if axis is None:
        result = func(arr)
        if function in ['mean', 'std']:
            result = round(float(result), 2)
    else:
        result = func(arr, axis=axis)
        if function in ['mean', 'std']:
            result = np.round(result, 2)

    axis_code = '' if axis is None else f', axis={axis}'
    code = f"""import numpy as np

np.random.seed({seed})
arr = np.random.randint(0, 100, ({rows}, {cols}))
print("陣列:")
print(arr)

# {func_name} ({axis_desc})
result = np.{func_name}(arr{axis_code})
print(f"\\nnp.{func_name}(arr{axis_code}) = {{result}}")"""

    extra_info = {
        'function': func_name,
        'axis': axis,
        'result_type': 'scalar' if axis is None else 'array'
    }

    return arr, result, code, extra_info


def process_sort_search(params, img=None):
    operation = params.get('operation', 'sort')
    axis_str = params.get('axis', 'none')
    rows = int(params.get('rows', 4))
    cols = int(params.get('cols', 5))
    threshold = int(params.get('threshold', 15))
    seed = int(params.get('seed', 42))

    np.random.seed(seed)
    arr = np.random.randint(0, 30, (rows, cols))

    axis = None if axis_str == 'none' else int(axis_str)

    if operation == 'sort':
        if axis is None:
            result = np.sort(arr.flatten())
        else:
            result = np.sort(arr, axis=axis)
        axis_code = '' if axis is None else f', axis={axis}'
        code = f"""import numpy as np

np.random.seed({seed})
arr = np.random.randint(0, 30, ({rows}, {cols}))
print("原始陣列:")
print(arr)

# 排序
result = np.sort(arr{'.flatten()' if axis is None else ''}{axis_code})
print("\\n排序後:")
print(result)"""
        extra_info = {'operation': 'sort'}

    elif operation == 'argsort':
        flat_arr = arr.flatten()
        result = np.argsort(flat_arr)
        code = f"""import numpy as np

np.random.seed({seed})
arr = np.random.randint(0, 30, ({rows}, {cols}))
flat = arr.flatten()
print("攤平陣列:")
print(flat)

# 排序索引
indices = np.argsort(flat)
print("\\nargsort 結果（排序後的索引）:")
print(indices)
print("\\n驗證：flat[indices] =")
print(flat[indices])"""
        extra_info = {'operation': 'argsort', 'flat': flat_arr.tolist()}

    elif operation == 'where':
        condition = arr > threshold
        indices = np.where(condition)
        result = {'rows': indices[0].tolist(), 'cols': indices[1].tolist()}
        code = f"""import numpy as np

np.random.seed({seed})
arr = np.random.randint(0, 30, ({rows}, {cols}))
print("陣列:")
print(arr)

# 尋找大於 {threshold} 的位置
rows, cols = np.where(arr > {threshold})
print(f"\\n大於 {threshold} 的位置:")
print(f"列索引: {{rows}}")
print(f"欄索引: {{cols}}")
print(f"\\n對應的值: {{arr[arr > {threshold}]}}")"""

        highlights = list(zip(indices[0].tolist(), indices[1].tolist()))
        extra_info = {'operation': 'where', 'highlight': highlights, 'threshold': threshold}

    elif operation == 'unique':
        flat_arr = arr.flatten()
        result = np.unique(flat_arr)
        code = f"""import numpy as np

np.random.seed({seed})
arr = np.random.randint(0, 30, ({rows}, {cols}))
print("原始陣列:")
print(arr)

# 找出唯一值
unique = np.unique(arr)
print(f"\\n唯一值 ({{len(unique)}} 個):")
print(unique)"""
        extra_info = {'operation': 'unique'}

    else:
        result = arr
        code = ""
        extra_info = {}

    return arr, result, code, extra_info


def process_image_as_array(params, img=None):
    if img is None:
        # 建立示範用的小圖
        demo_img = np.zeros((4, 5, 3), dtype=np.uint8)
        demo_img[0, :, 2] = 255  # 第一列紅色
        demo_img[1, :, 1] = 255  # 第二列綠色
        demo_img[2, :, 0] = 255  # 第三列藍色
        demo_img[3, :] = [128, 128, 128]  # 第四列灰色
        img = demo_img

    code = f"""import cv2
import numpy as np

# 讀取圖片
img = cv2.imread('image.jpg')

# 圖片就是 NumPy 陣列！
print(f"型態: {{type(img)}}")
print(f"形狀 (shape): {{img.shape}}")
print(f"資料類型 (dtype): {{img.dtype}}")
print(f"總元素數 (size): {{img.size}}")

# shape 解讀：
# ({img.shape[0]}, {img.shape[1]}, {img.shape[2]})
#   ↑       ↑       ↑
#  高度    寬度   通道數 (BGR)"""

    extra_info = {
        'shape': img.shape,
        'dtype': str(img.dtype),
        'size': img.size,
        'is_image': True
    }

    return img, img, code, extra_info


def process_channel_slice(params, img=None):
    if img is None:
        return None, None, "# 需要上傳圖片", {'error': '需要上傳圖片'}

    channel = params.get('channel', 'b')
    show_gray = params.get('show_gray', False)

    result = img.copy()

    if channel == 'b':
        if show_gray:
            result = img[:, :, 0]
        else:
            result = np.zeros_like(img)
            result[:, :, 0] = img[:, :, 0]
        code_slice = "img[:, :, 0]"
        desc = "B 通道"
    elif channel == 'g':
        if show_gray:
            result = img[:, :, 1]
        else:
            result = np.zeros_like(img)
            result[:, :, 1] = img[:, :, 1]
        code_slice = "img[:, :, 1]"
        desc = "G 通道"
    elif channel == 'r':
        if show_gray:
            result = img[:, :, 2]
        else:
            result = np.zeros_like(img)
            result[:, :, 2] = img[:, :, 2]
        code_slice = "img[:, :, 2]"
        desc = "R 通道"
    elif channel == 'no_b':
        result[:, :, 0] = 0
        code_slice = "img[:, :, 0] = 0"
        desc = "移除 B（黃色調）"
    elif channel == 'no_g':
        result[:, :, 1] = 0
        code_slice = "img[:, :, 1] = 0"
        desc = "移除 G（紫色調）"
    elif channel == 'no_r':
        result[:, :, 2] = 0
        code_slice = "img[:, :, 2] = 0"
        desc = "移除 R（青色調）"
    else:
        code_slice = "img"
        desc = "原圖"

    code = f"""import cv2
import numpy as np

img = cv2.imread('image.jpg')

# {desc}
# OpenCV 使用 BGR 順序：
#   img[:, :, 0] → B (藍色)
#   img[:, :, 1] → G (綠色)
#   img[:, :, 2] → R (紅色)

result = {code_slice}
cv2.imshow('Result', result)"""

    extra_info = {
        'channel': channel,
        'is_image': True
    }

    return img, result, code, extra_info


def process_roi_crop(params, img=None):
    if img is None:
        return None, None, "# 需要上傳圖片", {'error': '需要上傳圖片'}

    x_start = int(params.get('x_start', 20))
    x_end = int(params.get('x_end', 80))
    y_start = int(params.get('y_start', 20))
    y_end = int(params.get('y_end', 80))

    h, w = img.shape[:2]

    x1 = int(w * x_start / 100)
    x2 = int(w * x_end / 100)
    y1 = int(h * y_start / 100)
    y2 = int(h * y_end / 100)

    # 確保有效範圍
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    roi = img[y1:y2, x1:x2]

    code = f"""import cv2
import numpy as np

img = cv2.imread('image.jpg')
h, w = img.shape[:2]  # {h}, {w}

# 定義 ROI 範圍（注意：y 在前，x 在後）
y1, y2 = {y1}, {y2}
x1, x2 = {x1}, {x2}

# 用切片裁切 ROI
roi = img[y1:y2, x1:x2]
# 等於 img[{y1}:{y2}, {x1}:{x2}]

print(f"原圖大小: {{img.shape}}")
print(f"ROI 大小: {{roi.shape}}")"""

    extra_info = {
        'roi_bounds': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
        'is_image': True
    }

    return img, roi, code, extra_info


def process_pixel_math(params, img=None):
    if img is None:
        return None, None, "# 需要上傳圖片", {'error': '需要上傳圖片'}

    operation = params.get('operation', 'add')
    value = int(params.get('value', 50))
    gamma = float(params.get('gamma', 1.0))

    img_float = img.astype(np.float32)

    if operation == 'add':
        result = np.clip(img_float + value, 0, 255).astype(np.uint8)
        code = f"""import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 加法調整亮度（需要處理溢位）
img_float = img.astype(np.float32)
result = np.clip(img_float + {value}, 0, 255).astype(np.uint8)

# np.clip 確保值在 0~255 範圍內"""

    elif operation == 'sub':
        result = np.clip(img_float - value, 0, 255).astype(np.uint8)
        code = f"""import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 減法降低亮度
img_float = img.astype(np.float32)
result = np.clip(img_float - {abs(value)}, 0, 255).astype(np.uint8)"""

    elif operation == 'mul':
        factor = 1 + value / 100
        result = np.clip(img_float * factor, 0, 255).astype(np.uint8)
        code = f"""import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 乘法調整對比
factor = {factor:.2f}
img_float = img.astype(np.float32)
result = np.clip(img_float * factor, 0, 255).astype(np.uint8)"""

    elif operation == 'div':
        factor = 1 + abs(value) / 100
        result = np.clip(img_float / factor, 0, 255).astype(np.uint8)
        code = f"""import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 除法降低對比
factor = {factor:.2f}
img_float = img.astype(np.float32)
result = np.clip(img_float / factor, 0, 255).astype(np.uint8)"""

    elif operation == 'invert':
        result = 255 - img
        code = f"""import cv2
import numpy as np

img = cv2.imread('image.jpg')

# 反轉（負片效果）
result = 255 - img

# 或使用 cv2.bitwise_not(img)"""

    elif operation == 'power':
        normalized = img_float / 255.0
        result = (np.power(normalized, gamma) * 255).astype(np.uint8)
        code = f"""import cv2
import numpy as np

img = cv2.imread('image.jpg')

# Gamma 校正
gamma = {gamma}
normalized = img.astype(np.float32) / 255.0
result = (np.power(normalized, gamma) * 255).astype(np.uint8)

# gamma < 1: 變亮
# gamma > 1: 變暗
# gamma = 1: 不變"""

    else:
        result = img
        code = "# 未知操作"

    extra_info = {
        'operation': operation,
        'is_image': True
    }

    return img, result, code, extra_info


# 處理器映射
processors = {
    'array_create': process_array_create,
    'array_random': process_array_random,
    'array_special': process_array_special,
    'indexing': process_indexing,
    'slicing': process_slicing,
    'fancy_indexing': process_fancy_indexing,
    'reshape': process_reshape,
    'transpose': process_transpose,
    'stack_split': process_stack_split,
    'element_wise': process_element_wise,
    'broadcasting': process_broadcasting,
    'dot_product': process_dot_product,
    'statistics': process_statistics,
    'sort_search': process_sort_search,
    'image_as_array': process_image_as_array,
    'channel_slice': process_channel_slice,
    'roi_crop': process_roi_crop,
    'pixel_math': process_pixel_math
}
