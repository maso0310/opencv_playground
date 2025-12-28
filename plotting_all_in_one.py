# -*- coding: utf-8 -*-
"""
Python 繪圖全套範例模組
整合 Matplotlib, Seaborn, Plotly 等主流繪圖套件
提供完整的繪圖學習範例
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import io
import base64

# 設定中文字型
rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False

def fig_to_base64(fig):
    """將 matplotlib 圖表轉換為 base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{img_base64}'

# ========== 完整繪圖範例定義 ==========

PLOTTING_EXAMPLES = {
    # ========== Matplotlib 基礎 ==========
    'mpl_line': {
        'name': 'Matplotlib 折線圖',
        'category': 'Matplotlib 基礎',
        'library': 'matplotlib',
        'description': '''最基本的折線圖，用於顯示數據趨勢。

✨ 主要用途：
• 時間序列數據可視化
• 展示連續數據的變化
• 比較多組數據趨勢

📚 知識點：
• plt.plot() 基本用法
• 線條樣式設定 (linestyle, linewidth, color)
• 標記點設定 (marker, markersize)
• 圖表標題和標籤
• 圖例 (legend) 的使用
• 網格線設定 (grid)''',
        'params': [
            {'name': 'num_lines', 'label': '線條數量', 'type': 'slider', 'min': 1, 'max': 5, 'step': 1, 'default': 2},
            {'name': 'points', 'label': '數據點數', 'type': 'slider', 'min': 50, 'max': 200, 'step': 10, 'default': 100},
            {'name': 'show_markers', 'label': '顯示標記點', 'type': 'checkbox', 'default': False}
        ]
    },

    'mpl_scatter': {
        'name': 'Matplotlib 散點圖',
        'category': 'Matplotlib 基礎',
        'library': 'matplotlib',
        'description': '''散點圖用於顯示兩個變數之間的關係。

✨ 主要用途：
• 探索變數間的相關性
• 識別離群值
• 分類數據可視化

📚 知識點：
• plt.scatter() 用法
• 點大小 (s) 和顏色 (c) 設定
• 顏色映射 (cmap) 的使用
• 透明度 (alpha) 處理重疊
• 添加色條 (colorbar)
• 迴歸線繪製''',
        'params': [
            {'name': 'num_points', 'label': '數據點數', 'type': 'slider', 'min': 50, 'max': 500, 'step': 50, 'default': 200},
            {'name': 'correlation', 'label': '相關性', 'type': 'slider', 'min': -1, 'max': 1, 'step': 0.1, 'default': 0.7}
        ]
    },

    'mpl_bar': {
        'name': 'Matplotlib 長條圖',
        'category': 'Matplotlib 基礎',
        'library': 'matplotlib',
        'description': '''長條圖用於比較不同類別的數值。

✨ 主要用途：
• 比較不同類別的數量
• 展示排名
• 顯示分類數據分布

📚 知識點：
• plt.bar() 和 plt.barh()
• 長條顏色和邊框設定
• 數值標註
• 分組長條圖 (grouped bar)
• 堆疊長條圖 (stacked bar)''',
        'params': [
            {'name': 'num_bars', 'label': '長條數量', 'type': 'slider', 'min': 3, 'max': 12, 'step': 1, 'default': 6},
            {'name': 'orientation', 'label': '方向', 'type': 'select',
             'options': [{'value': 'vertical', 'label': '垂直'}, {'value': 'horizontal', 'label': '水平'}],
             'default': 'vertical'}
        ]
    },

    'mpl_pie': {
        'name': 'Matplotlib 圓餅圖',
        'category': 'Matplotlib 基礎',
        'library': 'matplotlib',
        'description': '''圓餅圖用於顯示各部分佔整體的比例。

✨ 主要用途：
• 顯示構成比例
• 市場佔有率
• 預算分配

📚 知識點：
• plt.pie() 用法
• autopct 顯示百分比
• startangle 起始角度
• explode 突出顯示
• 圓環圖 (donut chart)''',
        'params': [
            {'name': 'num_slices', 'label': '扇形數量', 'type': 'slider', 'min': 3, 'max': 8, 'step': 1, 'default': 5},
            {'name': 'explode', 'label': '突出最大扇形', 'type': 'checkbox', 'default': True}
        ]
    },

    'mpl_hist': {
        'name': 'Matplotlib 直方圖',
        'category': 'Matplotlib 基礎',
        'library': 'matplotlib',
        'description': '''直方圖用於顯示數據的分布情況。

✨ 主要用途：
• 分析數據分布
• 檢測分布形狀
• 質量控制

📚 知識點：
• plt.hist() 用法
• bins 分組設定
• density 密度顯示
• histtype 類型選擇
• 多組數據比較''',
        'params': [
            {'name': 'distribution', 'label': '分布類型', 'type': 'select',
             'options': [
                 {'value': 'normal', 'label': '常態分布'},
                 {'value': 'uniform', 'label': '均勻分布'},
                 {'value': 'exponential', 'label': '指數分布'}
             ], 'default': 'normal'},
            {'name': 'bins', 'label': '分組數', 'type': 'slider', 'min': 10, 'max': 50, 'step': 5, 'default': 30}
        ]
    },

    # ========== Matplotlib 進階 ==========
    'mpl_subplots': {
        'name': 'Matplotlib 子圖',
        'category': 'Matplotlib 進階',
        'library': 'matplotlib',
        'description': '''在一個圖表中顯示多個子圖。

✨ 主要用途：
• 同時展示多個相關圖表
• 比較不同視角的數據
• 綜合報告

📚 知識點：
• plt.subplots() 創建子圖
• fig, axes 物件導向介面
• 子圖布局設定
• 共享軸 (sharex, sharey)
• tight_layout() 調整間距''',
        'params': [
            {'name': 'rows', 'label': '行數', 'type': 'slider', 'min': 1, 'max': 3, 'step': 1, 'default': 2},
            {'name': 'cols', 'label': '列數', 'type': 'slider', 'min': 1, 'max': 3, 'step': 1, 'default': 2}
        ]
    },

    'mpl_heatmap': {
        'name': 'Matplotlib 熱力圖',
        'category': 'Matplotlib 進階',
        'library': 'matplotlib',
        'description': '''使用顏色深淺表示數值大小。

✨ 主要用途：
• 相關性矩陣可視化
• 混淆矩陣
• 時間序列模式

📚 知識點：
• plt.imshow() 或 pcolormesh()
• 顏色映射 (cmap) 選擇
• colorbar 色條
• 數值標註
• 插值方法''',
        'params': [
            {'name': 'size', 'label': '矩陣大小', 'type': 'slider', 'min': 5, 'max': 15, 'step': 1, 'default': 10},
            {'name': 'cmap', 'label': '色彩映射', 'type': 'select',
             'options': [
                 {'value': 'viridis', 'label': 'Viridis'},
                 {'value': 'coolwarm', 'label': 'Coolwarm'},
                 {'value': 'YlOrRd', 'label': 'YlOrRd'}
             ], 'default': 'viridis'}
        ]
    },

    'mpl_3d': {
        'name': 'Matplotlib 3D圖表',
        'category': 'Matplotlib 進階',
        'library': 'matplotlib',
        'description': '''三維數據可視化。

✨ 主要用途：
• 展示三維數據
• 曲面圖
• 3D散點圖

📚 知識點：
• Axes3D 使用
• plot_surface() 曲面圖
• scatter3D() 3D散點
• 視角設定
• wireframe 線框圖''',
        'params': [
            {'name': 'plot_type', 'label': '圖表類型', 'type': 'select',
             'options': [
                 {'value': 'surface', 'label': '曲面圖'},
                 {'value': 'wireframe', 'label': '線框圖'},
                 {'value': 'scatter', 'label': '散點圖'}
             ], 'default': 'surface'}
        ]
    },

    # ========== Seaborn ==========
    'sns_dist': {
        'name': 'Seaborn 分布圖',
        'category': 'Seaborn',
        'library': 'seaborn',
        'description': '''Seaborn 是基於 Matplotlib 的高級繪圖庫，提供更美觀的預設樣式。

✨ 主要用途：
• 快速探索數據分布
• 美觀的統計圖表
• 自動化樣式設定

📚 知識點：
• sns.histplot() 直方圖
• sns.kdeplot() 核密度估計
• sns.violinplot() 小提琴圖
• sns.boxplot() 盒鬚圖
• 主題設定 (set_theme)''',
        'params': [
            {'name': 'plot_type', 'label': '圖表類型', 'type': 'select',
             'options': [
                 {'value': 'hist', 'label': '直方圖'},
                 {'value': 'kde', 'label': '核密度估計'},
                 {'value': 'box', 'label': '盒鬚圖'},
                 {'value': 'violin', 'label': '小提琴圖'}
             ], 'default': 'hist'}
        ]
    },

    'sns_regression': {
        'name': 'Seaborn 迴歸圖',
        'category': 'Seaborn',
        'library': 'seaborn',
        'description': '''帶有迴歸線的散點圖。

✨ 主要用途：
• 探索變數間關係
• 迴歸分析
• 趨勢線擬合

📚 知識點：
• sns.regplot() 簡單迴歸
• sns.lmplot() 高級迴歸
• 信賴區間顯示
• 多項式迴歸
• 分組迴歸''',
        'params': [
            {'name': 'order', 'label': '多項式階數', 'type': 'slider', 'min': 1, 'max': 3, 'step': 1, 'default': 1}
        ]
    },

    'sns_heatmap': {
        'name': 'Seaborn 相關性熱力圖',
        'category': 'Seaborn',
        'library': 'seaborn',
        'description': '''比 Matplotlib 更方便的熱力圖。

✨ 主要用途：
• 相關性矩陣
• 數據表格可視化
• 特徵工程

📚 知識點：
• sns.heatmap() 用法
• annot 數值標註
• fmt 格式設定
• mask 遮罩使用
• cbar_kws 色條設定''',
        'params': [
            {'name': 'annot', 'label': '顯示數值', 'type': 'checkbox', 'default': True}
        ]
    },

    # ========== Plotly (互動式) ==========
    'plotly_line': {
        'name': 'Plotly 互動式折線圖',
        'category': 'Plotly 互動式',
        'library': 'plotly',
        'description': '''Plotly 提供互動式圖表，用戶可以縮放、平移、懸停查看數據。

✨ 主要用途：
• 互動式數據探索
• 網頁展示
• 儀表板

📚 知識點：
• plotly.graph_objects
• go.Scatter() 散點/折線
• update_layout() 布局設定
• hover_template 懸停資訊
• 導出為 HTML''',
        'params': [
            {'name': 'num_lines', 'label': '線條數量', 'type': 'slider', 'min': 1, 'max': 5, 'step': 1, 'default': 3}
        ]
    },

    'plotly_3d': {
        'name': 'Plotly 3D曲面圖',
        'category': 'Plotly 互動式',
        'library': 'plotly',
        'description': '''互動式3D可視化。

✨ 主要用途：
• 三維數據探索
• 曲面分析
• 地形圖

📚 知識點：
• go.Surface() 曲面圖
• go.Scatter3d() 3D散點
• 互動式旋轉
• 相機視角設定''',
        'params': []
    }
}


def get_all_plotting_effects():
    """取得所有繪圖效果"""
    return PLOTTING_EXAMPLES


# ========== 處理函數 ==========

def process_plotting(effect, params):
    """處理繪圖請求"""

    try:
        effect_info = PLOTTING_EXAMPLES.get(effect, {})
        library = effect_info.get('library', 'matplotlib')

        # Matplotlib 圖表
        if library == 'matplotlib':
            if effect == 'mpl_line':
                return _generate_mpl_line(params)
            elif effect == 'mpl_scatter':
                return _generate_mpl_scatter(params)
            elif effect == 'mpl_bar':
                return _generate_mpl_bar(params)
            elif effect == 'mpl_pie':
                return _generate_mpl_pie(params)
            elif effect == 'mpl_hist':
                return _generate_mpl_hist(params)
            elif effect == 'mpl_subplots':
                return _generate_mpl_subplots(params)
            elif effect == 'mpl_heatmap':
                return _generate_mpl_heatmap(params)
            elif effect == 'mpl_3d':
                return _generate_mpl_3d(params)

        # Seaborn 圖表
        elif library == 'seaborn':
            if effect == 'sns_dist':
                return _generate_sns_dist(params)
            elif effect == 'sns_regression':
                return _generate_sns_regression(params)
            elif effect == 'sns_heatmap':
                return _generate_sns_heatmap(params)

        # Plotly 圖表
        elif library == 'plotly':
            if effect == 'plotly_line':
                return _generate_plotly_line(params)
            elif effect == 'plotly_3d':
                return _generate_plotly_3d(params)

        return {'success': False, 'error': '尚未實現此圖表'}

    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# ========== Matplotlib 圖表生成函數 ==========

def _generate_mpl_line(params):
    """生成折線圖"""
    num_lines = int(params.get('num_lines', 2))
    points = int(params.get('points', 100))
    show_markers = params.get('show_markers', False)

    x = np.linspace(0, 10, points)
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, num_lines))
    for i in range(num_lines):
        y = np.sin(x + i * np.pi/num_lines) + np.random.randn(points) * 0.1
        if show_markers:
            ax.plot(x, y, color=colors[i], marker='o', markersize=3,
                   linewidth=2, label=f'數據線 {i+1}', alpha=0.8)
        else:
            ax.plot(x, y, color=colors[i], linewidth=2,
                   label=f'數據線 {i+1}', alpha=0.8)

    ax.set_xlabel('X 軸', fontsize=12)
    ax.set_ylabel('Y 軸', fontsize=12)
    ax.set_title('Matplotlib 折線圖範例', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    code = f"""import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, {points})
fig, ax = plt.subplots(figsize=(10, 6))

for i in range({num_lines}):
    y = np.sin(x + i * np.pi/{num_lines}) + np.random.randn({points}) * 0.1
    ax.plot(x, y, linewidth=2, label=f'數據線 {{i+1}}')

ax.set_xlabel('X 軸')
ax.set_ylabel('Y 軸')
ax.set_title('Matplotlib 折線圖範例')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()"""

    return {
        'success': True,
        'image': fig_to_base64(fig),
        'code': code
    }


def _generate_mpl_scatter(params):
    """生成散點圖"""
    num_points = int(params.get('num_points', 200))
    correlation = float(params.get('correlation', 0.7))

    # 生成相關數據
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    x, y = np.random.multivariate_normal(mean, cov, num_points).T

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x, y, s=50, alpha=0.6, c='steelblue',
                        edgecolors='navy', linewidth=0.5)

    # 添加迴歸線
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2,
            label=f'迴歸線: y={z[0]:.2f}x+{z[1]:.2f}')

    ax.set_xlabel('X 變數', fontsize=12)
    ax.set_ylabel('Y 變數', fontsize=12)
    ax.set_title(f'散點圖 (相關性 ≈ {correlation})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    code = f"""import matplotlib.pyplot as plt
import numpy as np

# 生成相關數據
mean = [0, 0]
cov = [[1, {correlation}], [{correlation}, 1]]
x, y = np.random.multivariate_normal(mean, cov, {num_points}).T

plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=50, alpha=0.6)

# 添加迴歸線
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, y)", "r--", alpha=0.8, label=f'迴歸線')

plt.xlabel('X 變數')
plt.ylabel('Y 變數')
plt.title(f'散點圖 (相關性 ≈ {correlation})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""

    return {
        'success': True,
        'image': fig_to_base64(fig),
        'code': code
    }


def _generate_mpl_bar(params):
    """生成長條圖"""
    num_bars = int(params.get('num_bars', 6))
    orientation = params.get('orientation', 'vertical')

    categories = [f'類別{i+1}' for i in range(num_bars)]
    values = np.random.randint(20, 100, num_bars)

    fig, ax = plt.subplots(figsize=(10, 6))

    if orientation == 'vertical':
        bars = ax.bar(categories, values, color='steelblue',
                     edgecolor='black', linewidth=1.2, alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        ax.set_ylabel('數值', fontsize=12)
    else:
        bars = ax.barh(categories, values, color='steelblue',
                      edgecolor='black', linewidth=1.2, alpha=0.8)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{int(width)}',
                   ha='left', va='center', fontsize=10)
        ax.set_xlabel('數值', fontsize=12)

    ax.set_title('Matplotlib 長條圖範例', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y' if orientation == 'vertical' else 'x')

    code = f"""import matplotlib.pyplot as plt
import numpy as np

categories = {categories}
values = {list(values)}

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='steelblue', edgecolor='black', alpha=0.8)
plt.ylabel('數值')
plt.title('Matplotlib 長條圖範例')
plt.grid(True, alpha=0.3)
plt.show()"""

    return {
        'success': True,
        'image': fig_to_base64(fig),
        'code': code
    }


def _generate_mpl_pie(params):
    """生成圓餅圖"""
    num_slices = int(params.get('num_slices', 5))
    explode_max = params.get('explode', True)

    labels = [f'類別{i+1}' for i in range(num_slices)]
    sizes = np.random.randint(10, 50, num_slices)

    explode = [0] * num_slices
    if explode_max:
        max_idx = np.argmax(sizes)
        explode[max_idx] = 0.1

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
          startangle=90, explode=explode, shadow=True)
    ax.set_title('Matplotlib 圓餅圖範例', fontsize=14, fontweight='bold')

    code = f"""import matplotlib.pyplot as plt
import numpy as np

labels = {labels}
sizes = {list(sizes)}

plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Matplotlib 圓餅圖範例')
plt.show()"""

    return {
        'success': True,
        'image': fig_to_base64(fig),
        'code': code
    }


def _generate_mpl_hist(params):
    """生成直方圖"""
    distribution = params.get('distribution', 'normal')
    bins = int(params.get('bins', 30))

    if distribution == 'normal':
        data = np.random.randn(1000)
        title = '常態分布'
    elif distribution == 'uniform':
        data = np.random.uniform(-3, 3, 1000)
        title = '均勻分布'
    else:  # exponential
        data = np.random.exponential(1, 1000)
        title = '指數分布'

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('數值', fontsize=12)
    ax.set_ylabel('頻率', fontsize=12)
    ax.set_title(f'Matplotlib 直方圖 - {title}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    code = f"""import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)  # {title}

plt.figure(figsize=(10, 6))
plt.hist(data, bins={bins}, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('數值')
plt.ylabel('頻率')
plt.title('Matplotlib 直方圖 - {title}')
plt.grid(True, alpha=0.3)
plt.show()"""

    return {
        'success': True,
        'image': fig_to_base64(fig),
        'code': code
    }


def _generate_mpl_subplots(params):
    """生成子圖"""
    rows = int(params.get('rows', 2))
    cols = int(params.get('cols', 2))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.suptitle('Matplotlib 子圖範例', fontsize=16, fontweight='bold')

    # 確保 axes 是二維陣列
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    plot_types = ['line', 'scatter', 'bar', 'hist']

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            plot_type = plot_types[(i * cols + j) % len(plot_types)]

            if plot_type == 'line':
                x = np.linspace(0, 10, 100)
                ax.plot(x, np.sin(x + (i*cols+j)))
                ax.set_title(f'折線圖 ({i+1},{j+1})')
            elif plot_type == 'scatter':
                x, y = np.random.randn(2, 50)
                ax.scatter(x, y)
                ax.set_title(f'散點圖 ({i+1},{j+1})')
            elif plot_type == 'bar':
                ax.bar(range(5), np.random.randint(1, 10, 5))
                ax.set_title(f'長條圖 ({i+1},{j+1})')
            else:  # hist
                ax.hist(np.random.randn(100), bins=20)
                ax.set_title(f'直方圖 ({i+1},{j+1})')

            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    code = f"""import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots({rows}, {cols}, figsize=(12, 8))
fig.suptitle('Matplotlib 子圖範例')

# 繪製不同類型的圖表
# axes[i, j].plot(...)  # 折線圖
# axes[i, j].scatter(...)  # 散點圖
# axes[i, j].bar(...)  # 長條圖
# axes[i, j].hist(...)  # 直方圖

plt.tight_layout()
plt.show()"""

    return {
        'success': True,
        'image': fig_to_base64(fig),
        'code': code
    }


def _generate_mpl_heatmap(params):
    """生成熱力圖"""
    size = int(params.get('size', 10))
    cmap = params.get('cmap', 'viridis')

    data = np.random.rand(size, size)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, label='數值')

    ax.set_xlabel('X 軸', fontsize=12)
    ax.set_ylabel('Y 軸', fontsize=12)
    ax.set_title('Matplotlib 熱力圖範例', fontsize=14, fontweight='bold')

    code = f"""import matplotlib.pyplot as plt
import numpy as np

data = np.random.rand({size}, {size})

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='{cmap}', aspect='auto')
plt.colorbar(label='數值')
plt.xlabel('X 軸')
plt.ylabel('Y 軸')
plt.title('Matplotlib 熱力圖範例')
plt.show()"""

    return {
        'success': True,
        'image': fig_to_base64(fig),
        'code': code
    }


def _generate_mpl_3d(params):
    """生成3D圖表"""
    from mpl_toolkits.mplot3d import Axes3D

    plot_type = params.get('plot_type', 'surface')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if plot_type == 'surface':
        X = np.linspace(-5, 5, 50)
        Y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5)
        ax.set_title('3D 曲面圖')

    elif plot_type == 'wireframe':
        X = np.linspace(-5, 5, 30)
        Y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.6)
        ax.set_title('3D 線框圖')

    else:  # scatter
        n = 200
        x = np.random.randn(n)
        y = np.random.randn(n)
        z = np.random.randn(n)
        colors = np.sqrt(x**2 + y**2 + z**2)
        ax.scatter(x, y, z, c=colors, cmap='viridis', s=50)
        ax.set_title('3D 散點圖')

    ax.set_xlabel('X 軸')
    ax.set_ylabel('Y 軸')
    ax.set_zlabel('Z 軸')

    code = f"""import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X 軸')
ax.set_ylabel('Y 軸')
ax.set_zlabel('Z 軸')
plt.show()"""

    return {
        'success': True,
        'image': fig_to_base64(fig),
        'code': code
    }


# ========== Seaborn 圖表生成函數 ==========

def _generate_sns_dist(params):
    """生成 Seaborn 分布圖"""
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid")

        plot_type = params.get('plot_type', 'hist')
        data = np.random.randn(500)

        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == 'hist':
            sns.histplot(data, kde=True, ax=ax)
            title = 'Seaborn 直方圖 + KDE'
        elif plot_type == 'kde':
            sns.kdeplot(data, fill=True, ax=ax)
            title = 'Seaborn 核密度估計'
        elif plot_type == 'box':
            sns.boxplot(x=data, ax=ax)
            title = 'Seaborn 盒鬚圖'
        else:  # violin
            sns.violinplot(x=data, ax=ax)
            title = 'Seaborn 小提琴圖'

        ax.set_title(title, fontsize=14, fontweight='bold')

        code = f"""import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="darkgrid")
data = np.random.randn(500)

plt.figure(figsize=(10, 6))
sns.{plot_type}plot(data, kde=True if '{plot_type}' == 'hist' else False)
plt.title('{title}')
plt.show()"""

        return {
            'success': True,
            'image': fig_to_base64(fig),
            'code': code
        }
    except ImportError:
        return {
            'success': False,
            'error': '請先安裝 seaborn 套件: pip install seaborn'
        }


def _generate_sns_regression(params):
    """生成 Seaborn 迴歸圖"""
    try:
        import seaborn as sns
        sns.set_theme(style="darkgrid")

        order = int(params.get('order', 1))

        x = np.random.rand(100) * 10
        y = 2 * x + np.random.randn(100) * 3

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=x, y=y, order=order, ax=ax)
        ax.set_title(f'Seaborn 迴歸圖 (多項式階數={order})', fontsize=14, fontweight='bold')
        ax.set_xlabel('X 變數')
        ax.set_ylabel('Y 變數')

        code = f"""import seaborn as sns
import numpy as np

sns.set_theme(style="darkgrid")
x = np.random.rand(100) * 10
y = 2 * x + np.random.randn(100) * 3

sns.regplot(x=x, y=y, order={order})
plt.xlabel('X 變數')
plt.ylabel('Y 變數')
plt.title('Seaborn 迴歸圖')
plt.show()"""

        return {
            'success': True,
            'image': fig_to_base64(fig),
            'code': code
        }
    except ImportError:
        return {
            'success': False,
            'error': '請先安裝 seaborn 套件: pip install seaborn'
        }


def _generate_sns_heatmap(params):
    """生成 Seaborn 熱力圖"""
    try:
        import seaborn as sns
        sns.set_theme()

        annot = params.get('annot', True)

        # 生成相關性矩陣
        data = np.random.randn(10, 10)
        df = pd.DataFrame(data, columns=[f'變數{i+1}' for i in range(10)])
        corr = df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=annot, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Seaborn 相關性熱力圖', fontsize=14, fontweight='bold')

        code = f"""import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme()
data = np.random.randn(10, 10)
df = pd.DataFrame(data, columns=[f'變數{{i+1}}' for i in range(10)])
corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot={annot}, cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Seaborn 相關性熱力圖')
plt.show()"""

        return {
            'success': True,
            'image': fig_to_base64(fig),
            'code': code
        }
    except ImportError:
        return {
            'success': False,
            'error': '請先安裝 seaborn 套件: pip install seaborn'
        }


# ========== Plotly 圖表生成函數 ==========

def _generate_plotly_line(params):
    """生成 Plotly 互動式折線圖"""
    try:
        import plotly.graph_objects as go

        num_lines = int(params.get('num_lines', 3))

        fig = go.Figure()

        x = np.linspace(0, 10, 100)
        for i in range(num_lines):
            y = np.sin(x + i * np.pi/num_lines) + np.random.randn(100) * 0.1
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                    name=f'數據線 {i+1}'))

        fig.update_layout(
            title='Plotly 互動式折線圖',
            xaxis_title='X 軸',
            yaxis_title='Y 軸',
            hovermode='x unified'
        )

        # 轉換為 HTML
        html = fig.to_html(include_plotlyjs='cdn', div_id='plotly-chart')

        code = f"""import plotly.graph_objects as go
import numpy as np

fig = go.Figure()

x = np.linspace(0, 10, 100)
for i in range({num_lines}):
    y = np.sin(x + i * np.pi/{num_lines}) + np.random.randn(100) * 0.1
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'數據線 {{i+1}}'))

fig.update_layout(
    title='Plotly 互動式折線圖',
    xaxis_title='X 軸',
    yaxis_title='Y 軸',
    hovermode='x unified'
)

fig.show()"""

        return {
            'success': True,
            'html': html,
            'code': code,
            'is_interactive': True
        }
    except ImportError:
        return {
            'success': False,
            'error': '請先安裝 plotly 套件: pip install plotly'
        }


def _generate_plotly_3d(params):
    """生成 Plotly 3D曲面圖"""
    try:
        import plotly.graph_objects as go

        X = np.linspace(-5, 5, 50)
        Y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(np.sqrt(X**2 + Y**2))

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

        fig.update_layout(
            title='Plotly 3D曲面圖',
            scene=dict(
                xaxis_title='X 軸',
                yaxis_title='Y 軸',
                zaxis_title='Z 軸'
            ),
            autosize=False,
            width=800,
            height=800
        )

        html = fig.to_html(include_plotlyjs='cdn', div_id='plotly-chart-3d')

        code = """import plotly.graph_objects as go
import numpy as np

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

fig.update_layout(
    title='Plotly 3D曲面圖',
    scene=dict(
        xaxis_title='X 軸',
        yaxis_title='Y 軸',
        zaxis_title='Z 軸'
    )
)

fig.show()"""

        return {
            'success': True,
            'html': html,
            'code': code,
            'is_interactive': True
        }
    except ImportError:
        return {
            'success': False,
            'error': '請先安裝 plotly 套件: pip install plotly'
        }
