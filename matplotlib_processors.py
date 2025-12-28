# -*- coding: utf-8 -*-
"""
Matplotlib 圖表處理函數
實際生成各種圖表的程式碼
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import io
import base64
from scipy import stats
from scipy.interpolate import make_interp_spline

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

def process_matplotlib_plot(effect, params):
    """處理 matplotlib 繪圖請求"""

    # ===== 基礎折線圖 =====
    if effect == 'line_basic':
        data_type = params.get('data_type', 'sine')
        points = int(params.get('points', 100))
        linewidth = float(params.get('linewidth', 2))
        show_markers = params.get('show_markers', False)
        show_grid = params.get('show_grid', True)

        x = np.linspace(0, 10, points)

        if data_type == 'sine':
            y = np.sin(x)
            title = '正弦波函數'
        elif data_type == 'linear':
            y = 0.5 * x + 2
            title = '線性增長'
        elif data_type == 'random':
            y = np.cumsum(np.random.randn(points)) / 10
            title = '隨機漫步'
        else:  # exponential
            y = np.exp(x / 5)
            title = '指數增長'

        fig, ax = plt.subplots(figsize=(10, 6))

        if show_markers:
            ax.plot(x, y, linewidth=linewidth, marker='o', markersize=4, label='數據線')
        else:
            ax.plot(x, y, linewidth=linewidth, label='數據線')

        ax.set_xlabel('X 軸', fontsize=12)
        ax.set_ylabel('Y 軸', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')

        code = f"""import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, {points})
y = {f'np.sin(x)' if data_type == 'sine' else f'0.5 * x + 2' if data_type == 'linear' else f'np.cumsum(np.random.randn({points})) / 10' if data_type == 'random' else 'np.exp(x / 5)'}

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth={linewidth}{', marker="o", markersize=4' if show_markers else ''}, label='數據線')
plt.xlabel('X 軸', fontsize=12)
plt.ylabel('Y 軸', fontsize=12)
plt.title('{title}', fontsize=14, fontweight='bold')
plt.legend()
{f"plt.grid(True, alpha=0.3, linestyle='--')" if show_grid else ''}
plt.show()"""

        return fig_to_base64(fig), code

    # ===== 多線比較圖 =====
    elif effect == 'line_multi':
        num_lines = int(params.get('num_lines', 3))
        noise_level = float(params.get('noise_level', 0.3))
        show_legend = params.get('show_legend', True)

        x = np.linspace(0, 10, 100)
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, num_lines))
        linestyles = ['-', '--', '-.', ':']

        for i in range(num_lines):
            base = np.sin(x + i * np.pi / num_lines)
            noise = np.random.randn(100) * noise_level
            y = base + noise

            ax.plot(x, y,
                   color=colors[i],
                   linestyle=linestyles[i % len(linestyles)],
                   linewidth=2,
                   label=f'數據系列 {i+1}',
                   alpha=0.8)

        ax.set_xlabel('X 軸', fontsize=12)
        ax.set_ylabel('Y 軸', fontsize=12)
        ax.set_title('多線比較圖', fontsize=14, fontweight='bold')
        if show_legend:
            ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        code = f"""import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.tab10(np.linspace(0, 1, {num_lines}))
linestyles = ['-', '--', '-.', ':']

for i in range({num_lines}):
    base = np.sin(x + i * np.pi / {num_lines})
    noise = np.random.randn(100) * {noise_level}
    y = base + noise

    ax.plot(x, y,
           color=colors[i],
           linestyle=linestyles[i % len(linestyles)],
           linewidth=2,
           label=f'數據系列 {{i+1}}',
           alpha=0.8)

ax.set_xlabel('X 軸', fontsize=12)
ax.set_ylabel('Y 軸', fontsize=12)
ax.set_title('多線比較圖', fontsize=14)
{'ax.legend(loc="best")' if show_legend else ''}
ax.grid(True, alpha=0.3)
plt.show()"""

        return fig_to_base64(fig), code

    # ===== 風格化折線圖 =====
    elif effect == 'line_styled':
        style = params.get('style', 'solid')
        marker = params.get('marker', 'o')
        markersize = int(params.get('markersize', 8))

        style_map = {
            'solid': '-',
            'dashed': '--',
            'dashdot': '-.',
            'dotted': ':'
        }

        x = np.linspace(0, 10, 50)
        y = np.sin(x) * np.exp(-x/10)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x, y,
               linestyle=style_map[style],
               marker=marker,
               markersize=markersize,
               linewidth=2.5,
               color='#2E86AB',
               markerfacecolor='#A23B72',
               markeredgecolor='#F18F01',
               markeredgewidth=2,
               label='風格化曲線')

        ax.set_xlabel('X 軸', fontsize=12)
        ax.set_ylabel('Y 軸', fontsize=12)
        ax.set_title('風格化折線圖範例', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#F5F5F5')

        code = f"""import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)
y = np.sin(x) * np.exp(-x/10)

plt.figure(figsize=(10, 6))
plt.plot(x, y,
        linestyle='{style_map[style]}',
        marker='{marker}',
        markersize={markersize},
        linewidth=2.5,
        color='#2E86AB',
        markerfacecolor='#A23B72',
        markeredgecolor='#F18F01',
        markeredgewidth=2,
        label='風格化曲線')

plt.xlabel('X 軸', fontsize=12)
plt.ylabel('Y 軸', fontsize=12)
plt.title('風格化折線圖範例', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
plt.gca().set_facecolor('#F5F5F5')
plt.show()"""

        return fig_to_base64(fig), code

    # ===== 基礎散點圖 =====
    elif effect == 'scatter_basic':
        num_points = int(params.get('num_points', 200))
        correlation = float(params.get('correlation', 0.7))
        point_size = int(params.get('point_size', 50))
        alpha = float(params.get('alpha', 0.6))

        # 生成相關數據
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        x, y = np.random.multivariate_normal(mean, cov, num_points).T

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(x, y,
                           s=point_size,
                           alpha=alpha,
                           c='steelblue',
                           edgecolors='navy',
                           linewidth=0.5)

        ax.set_xlabel('X 變數', fontsize=12)
        ax.set_ylabel('Y 變數', fontsize=12)
        ax.set_title(f'散點圖 (相關係數 ≈ {correlation})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

        # 添加迴歸線
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'迴歸線: y={z[0]:.2f}x+{z[1]:.2f}')
        ax.legend()

        code = f"""import matplotlib.pyplot as plt
import numpy as np

# 生成相關數據
mean = [0, 0]
cov = [[1, {correlation}], [{correlation}, 1]]
x, y = np.random.multivariate_normal(mean, cov, {num_points}).T

plt.figure(figsize=(10, 8))
plt.scatter(x, y,
           s={point_size},
           alpha={alpha},
           c='steelblue',
           edgecolors='navy',
           linewidth=0.5)

plt.xlabel('X 變數', fontsize=12)
plt.ylabel('Y 變數', fontsize=12)
plt.title(f'散點圖 (相關係數 ≈ {correlation})', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
plt.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

# 添加迴歸線
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'迴歸線')
plt.legend()
plt.show()"""

        return fig_to_base64(fig), code

    # ===== 彩色散點圖 =====
    elif effect == 'scatter_colored':
        colormap = params.get('colormap', 'viridis')
        show_colorbar = params.get('show_colorbar', True)

        num_points = 300
        x = np.random.randn(num_points)
        y = np.random.randn(num_points)
        colors = np.sqrt(x**2 + y**2)  # 距離作為顏色

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(x, y,
                           s=60,
                           c=colors,
                           cmap=colormap,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)

        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('距離原點', fontsize=11)

        ax.set_xlabel('X 軸', fontsize=12)
        ax.set_ylabel('Y 軸', fontsize=12)
        ax.set_title('彩色編碼散點圖', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        code = f"""import matplotlib.pyplot as plt
import numpy as np

num_points = 300
x = np.random.randn(num_points)
y = np.random.randn(num_points)
colors = np.sqrt(x**2 + y**2)  # 距離作為顏色

plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y,
                     s=60,
                     c=colors,
                     cmap='{colormap}',
                     alpha=0.7,
                     edgecolors='black',
                     linewidth=0.5)

{'plt.colorbar(scatter, label="距離原點")' if show_colorbar else ''}
plt.xlabel('X 軸', fontsize=12)
plt.ylabel('Y 軸', fontsize=12)
plt.title('彩色編碼散點圖', fontsize=14)
plt.grid(True, alpha=0.3)
plt.gca().set_aspect('equal')
plt.show()"""

        return fig_to_base64(fig), code

    # ===== 泡泡圖 =====
    elif effect == 'scatter_bubble':
        num_bubbles = int(params.get('num_bubbles', 50))
        size_range = int(params.get('size_range', 200))

        x = np.random.rand(num_bubbles) * 100
        y = np.random.rand(num_bubbles) * 100
        sizes = np.random.rand(num_bubbles) * size_range + 50
        colors = np.random.rand(num_bubbles)

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(x, y,
                           s=sizes,
                           c=colors,
                           cmap='Spectral',
                           alpha=0.6,
                           edgecolors='black',
                           linewidth=1.5)

        plt.colorbar(scatter, ax=ax, label='類別')

        ax.set_xlabel('X 變數 (例如: GDP)', fontsize=12)
        ax.set_ylabel('Y 變數 (例如: 預期壽命)', fontsize=12)
        ax.set_title('泡泡圖 (點大小=人口)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        code = f"""import matplotlib.pyplot as plt
import numpy as np

num_bubbles = {num_bubbles}
x = np.random.rand(num_bubbles) * 100
y = np.random.rand(num_bubbles) * 100
sizes = np.random.rand(num_bubbles) * {size_range} + 50
colors = np.random.rand(num_bubbles)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y,
                     s=sizes,
                     c=colors,
                     cmap='Spectral',
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=1.5)

plt.colorbar(scatter, label='類別')
plt.xlabel('X 變數 (例如: GDP)', fontsize=12)
plt.ylabel('Y 變數 (例如: 預期壽命)', fontsize=12)
plt.title('泡泡圖 (點大小=人口)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()"""

        return fig_to_base64(fig), code

    # ===== 基礎長條圖 =====
    elif effect == 'bar_basic':
        num_bars = int(params.get('num_bars', 6))
        orientation = params.get('orientation', 'vertical')
        show_values = params.get('show_values', True)
        colorful = params.get('colorful', False)

        categories = [f'項目{i+1}' for i in range(num_bars)]
        values = np.random.randint(20, 100, num_bars)

        fig, ax = plt.subplots(figsize=(10, 6))

        if colorful:
            colors = plt.cm.Set3(np.linspace(0, 1, num_bars))
        else:
            colors = 'steelblue'

        if orientation == 'vertical':
            bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.set_ylabel('數值', fontsize=12)
            ax.set_xlabel('類別', fontsize=12)
        else:  # horizontal
            bars = ax.barh(categories, values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
            if show_values:
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                           f'{int(width)}',
                           ha='left', va='center', fontsize=10, fontweight='bold', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
            ax.set_xlabel('數值', fontsize=12)
            ax.set_ylabel('類別', fontsize=12)

        ax.set_title('基礎長條圖', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y' if orientation == 'vertical' else 'x')

        code = f"""import matplotlib.pyplot as plt
import numpy as np

categories = {categories}
values = np.array({list(values)})

plt.figure(figsize=(10, 6))
{'bars = plt.bar(categories, values, color=plt.cm.Set3(np.linspace(0, 1, len(categories))), edgecolor="black", linewidth=1.2, alpha=0.8)' if colorful else 'bars = plt.bar(categories, values, color="steelblue", edgecolor="black", linewidth=1.2, alpha=0.8)'}

{f'''for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{{int(height)}}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')''' if show_values and orientation == 'vertical' else ''}

plt.ylabel('數值', fontsize=12)
plt.xlabel('類別', fontsize=12)
plt.title('基礎長條圖', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.show()"""

        return fig_to_base64(fig), code


    # 預設返回
    return None, "# 尚未實現此圖表類型"
