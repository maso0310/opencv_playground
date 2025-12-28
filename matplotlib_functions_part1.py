# -*- coding: utf-8 -*-
"""
Matplotlib 學習功能模組 - 第1部分：基礎圖表
提供各種 Matplotlib 基礎圖表的定義和處理函數
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI後端
import matplotlib.pyplot as plt
from matplotlib import rcParams
import io
import base64

# 設定中文字型
rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False

def fig_to_base64(fig):
    """將 matplotlib 圖表轉換為 base64 字串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{img_base64}'

# ===== Matplotlib 基礎圖表定義 =====

MATPLOTLIB_EFFECTS_PART1 = {
    # ===== 類別 1: 基礎線圖 =====
    'line_basic': {
        'name': '基礎折線圖',
        'category': '基礎線圖',
        'description': '''折線圖是最基本且常用的圖表類型，用於顯示數據隨時間或序列的變化趨勢。

主要用途：
• 顯示時間序列數據（股價、溫度變化）
• 展示連續數據的趨勢
• 比較多組數據的變化

基本語法：
plt.plot(x, y, label='標籤', color='顏色', linewidth=線寬, linestyle='線型')

常用參數：
• color: 'red', 'blue', '#FF5733' (十六進位)
• linestyle: '-'(實線), '--'(虛線), '-.'(點劃線), ':'(點線)
• marker: 'o'(圓點), 's'(方形), '^'(三角形)
• linewidth: 線條粗細''',
        'params': [
            {'name': 'data_type', 'label': '數據類型', 'type': 'select',
             'options': [
                 {'value': 'sine', 'label': '正弦波'},
                 {'value': 'linear', 'label': '線性增長'},
                 {'value': 'random', 'label': '隨機波動'},
                 {'value': 'exponential', 'label': '指數增長'}
             ], 'default': 'sine'},
            {'name': 'points', 'label': '數據點數', 'type': 'slider', 'min': 10, 'max': 200, 'step': 10, 'default': 100},
            {'name': 'linewidth', 'label': '線條粗細', 'type': 'slider', 'min': 1, 'max': 5, 'step': 0.5, 'default': 2},
            {'name': 'show_markers', 'label': '顯示標記點', 'type': 'checkbox', 'default': False},
            {'name': 'show_grid', 'label': '顯示網格', 'type': 'checkbox', 'default': True}
        ]
    },

    'line_multi': {
        'name': '多線比較圖',
        'category': '基礎線圖',
        'description': '''在同一圖表中繪製多條線，用於比較不同數據集的趨勢。

適用情境：
• 比較不同產品的銷售趨勢
• 顯示多個感測器的讀數
• 對比不同處理方法的效果

重要技巧：
• 使用不同顏色和線型區分
• 添加圖例說明 (legend)
• 適當調整線條粗細避免視覺混亂
• 使用 alpha 參數設置透明度''',
        'params': [
            {'name': 'num_lines', 'label': '線條數量', 'type': 'slider', 'min': 2, 'max': 5, 'step': 1, 'default': 3},
            {'name': 'noise_level', 'label': '噪聲程度', 'type': 'slider', 'min': 0, 'max': 1, 'step': 0.1, 'default': 0.3},
            {'name': 'show_legend', 'label': '顯示圖例', 'type': 'checkbox', 'default': True}
        ]
    },

    'line_styled': {
        'name': '風格化折線圖',
        'category': '基礎線圖',
        'description': '''展示各種線條風格和標記的組合，創建更美觀的圖表。

線型樣式：
• '-' : 實線
• '--' : 虛線
• '-.' : 點劃線
• ':' : 點線

標記樣式：
• 'o' : 圓形
• 's' : 方形
• '^' : 三角形
• 'D' : 菱形
• '*' : 星形
• '+' : 加號

顏色選擇：
• 單字元：'r' (紅), 'g' (綠), 'b' (藍), 'k' (黑)
• 全名：'red', 'blue', 'green'
• 十六進位：'#FF5733'
• RGB tuple：(0.1, 0.2, 0.5)''',
        'params': [
            {'name': 'style', 'label': '線條風格', 'type': 'select',
             'options': [
                 {'value': 'solid', 'label': '實線'},
                 {'value': 'dashed', 'label': '虛線'},
                 {'value': 'dashdot', 'label': '點劃線'},
                 {'value': 'dotted', 'label': '點線'}
             ], 'default': 'solid'},
            {'name': 'marker', 'label': '標記樣式', 'type': 'select',
             'options': [
                 {'value': 'o', 'label': '圓形 (o)'},
                 {'value': 's', 'label': '方形 (s)'},
                 {'value': '^', 'label': '三角形 (^)'},
                 {'value': 'D', 'label': '菱形 (D)'},
                 {'value': '*', 'label': '星形 (*)'}
             ], 'default': 'o'},
            {'name': 'markersize', 'label': '標記大小', 'type': 'slider', 'min': 3, 'max': 15, 'step': 1, 'default': 8}
        ]
    },

    # ===== 類別 2: 散點圖 =====
    'scatter_basic': {
        'name': '基礎散點圖',
        'category': '散點圖',
        'description': '''散點圖用於顯示兩個變數之間的關係，每個點代表一組觀測值。

主要用途：
• 探索變數間的相關性
• 識別離群值 (outliers)
• 展示數據分布模式
• 分類數據的可視化

基本語法：
plt.scatter(x, y, s=size, c=color, alpha=transparency, marker=shape)

重要參數：
• s: 點的大小（可以是陣列，每點不同大小）
• c: 顏色（可以是陣列，用於表示第三個維度）
• alpha: 透明度 (0-1)，處理重疊點
• cmap: 顏色映射（'viridis', 'plasma', 'rainbow'等）''',
        'params': [
            {'name': 'num_points', 'label': '數據點數', 'type': 'slider', 'min': 50, 'max': 500, 'step': 50, 'default': 200},
            {'name': 'correlation', 'label': '相關性', 'type': 'slider', 'min': -1, 'max': 1, 'step': 0.1, 'default': 0.7},
            {'name': 'point_size', 'label': '點大小', 'type': 'slider', 'min': 10, 'max': 100, 'step': 10, 'default': 50},
            {'name': 'alpha', 'label': '透明度', 'type': 'slider', 'min': 0.1, 'max': 1, 'step': 0.1, 'default': 0.6}
        ]
    },

    'scatter_colored': {
        'name': '彩色散點圖',
        'category': '散點圖',
        'description': '''使用顏色編碼第三個維度的數據，創建更具信息量的散點圖。

顏色映射 (Colormap)：
• Sequential: 'viridis', 'plasma', 'inferno'（用於連續數據）
• Diverging: 'coolwarm', 'RdYlBu'（用於有中心值的數據）
• Qualitative: 'tab10', 'Set3'（用於分類數據）

色條 (Colorbar)：
• 使用 plt.colorbar() 添加
• 顯示顏色對應的數值範圍
• 可自訂標籤和刻度

應用場景：
• 地理數據的海拔可視化
• 溫度分布圖
• 密度或權重的表示''',
        'params': [
            {'name': 'colormap', 'label': '色彩映射', 'type': 'select',
             'options': [
                 {'value': 'viridis', 'label': 'Viridis（藍綠漸變）'},
                 {'value': 'plasma', 'label': 'Plasma（紫橙漸變）'},
                 {'value': 'coolwarm', 'label': 'Coolwarm（藍紅對比）'},
                 {'value': 'rainbow', 'label': 'Rainbow（彩虹）'}
             ], 'default': 'viridis'},
            {'name': 'show_colorbar', 'label': '顯示色條', 'type': 'checkbox', 'default': True}
        ]
    },

    'scatter_bubble': {
        'name': '泡泡圖',
        'category': '散點圖',
        'description': '''泡泡圖是散點圖的延伸，使用點的大小表示第三個維度。

四個維度的表示：
• X 軸：第一個變數
• Y 軸：第二個變數
• 顏色：第三個變數
• 大小：第四個變數

應用實例：
• 人口統計：GDP vs 預期壽命，點大小=人口，顏色=地區
• 投資分析：風險 vs 報酬，點大小=投資額，顏色=產業
• 農業數據：溫度 vs 濕度，點大小=產量，顏色=作物類型

注意事項：
• 避免點過大導致重疊
• 使用透明度處理遮擋問題
• 適當縮放大小範圍''',
        'params': [
            {'name': 'num_bubbles', 'label': '泡泡數量', 'type': 'slider', 'min': 20, 'max': 100, 'step': 10, 'default': 50},
            {'name': 'size_range', 'label': '大小範圍', 'type': 'slider', 'min': 50, 'max': 500, 'step': 50, 'default': 200}
        ]
    },

    # ===== 類別 3: 長條圖 =====
    'bar_basic': {
        'name': '基礎長條圖',
        'category': '長條圖',
        'description': '''長條圖用於比較不同類別的數值，是最直觀的數據比較方式。

主要用途：
• 比較不同類別的數量或頻率
• 展示排名或優先順序
• 顯示分類數據的分布

基本語法：
plt.bar(x, height, width=0.8, color=colors, edgecolor='black')

plt.barh(y, width) # 水平長條圖

重要參數：
• width: 長條寬度（垂直長條圖）
• height: 長條高度（水平長條圖）
• color: 可以是單一顏色或顏色列表
• edgecolor: 邊框顏色
• alpha: 透明度''',
        'params': [
            {'name': 'num_bars', 'label': '長條數量', 'type': 'slider', 'min': 3, 'max': 12, 'step': 1, 'default': 6},
            {'name': 'orientation', 'label': '方向', 'type': 'select',
             'options': [
                 {'value': 'vertical', 'label': '垂直'},
                 {'value': 'horizontal', 'label': '水平'}
             ], 'default': 'vertical'},
            {'name': 'show_values', 'label': '顯示數值', 'type': 'checkbox', 'default': True},
            {'name': 'colorful', 'label': '彩色長條', 'type': 'checkbox', 'default': False}
        ]
    },

    'bar_grouped': {
        'name': '分組長條圖',
        'category': '長條圖',
        'description': '''分組長條圖用於比較多個類別在不同條件下的表現。

適用情境：
• 比較不同年份的各月銷售額
• 對比不同產品在各地區的銷量
• 展示多組實驗的結果比較

實作技巧：
• 計算每組的位置偏移
• 設定適當的長條寬度避免重疊
• 使用不同顏色區分組別
• 添加清晰的圖例

範例：
x = np.arange(len(categories))
width = 0.25
plt.bar(x - width, values1, width, label='組別1')
plt.bar(x, values2, width, label='組別2')
plt.bar(x + width, values3, width, label='組別3')''',
        'params': [
            {'name': 'num_categories', 'label': '類別數', 'type': 'slider', 'min': 3, 'max': 8, 'step': 1, 'default': 5},
            {'name': 'num_groups', 'label': '組別數', 'type': 'slider', 'min': 2, 'max': 4, 'step': 1, 'default': 3}
        ]
    },

    'bar_stacked': {
        'name': '堆疊長條圖',
        'category': '長條圖',
        'description': '''堆疊長條圖將多個數值堆疊在同一長條中，顯示總量和組成。

主要用途：
• 顯示整體和各部分的關係
• 比較不同類別的總量和組成
• 展示百分比構成

語法要點：
plt.bar(x, values1, label='部分1')
plt.bar(x, values2, bottom=values1, label='部分2')
plt.bar(x, values3, bottom=values1+values2, label='部分3')

應用實例：
• 預算分配（不同項目的支出）
• 能源構成（煤、油、天然氣、再生能源）
• 時間分配（工作、休息、娛樂）

注意事項：
• 不要堆疊太多層（建議≤5層）
• 使用對比鮮明的顏色
• 考慮使用百分比堆疊圖''',
        'params': [
            {'name': 'num_categories', 'label': '類別數', 'type': 'slider', 'min': 4, 'max': 10, 'step': 1, 'default': 6},
            {'name': 'num_parts', 'label': '堆疊層數', 'type': 'slider', 'min': 2, 'max': 5, 'step': 1, 'default': 3},
            {'name': 'percentage', 'label': '百分比模式', 'type': 'checkbox', 'default': False}
        ]
    }
}
