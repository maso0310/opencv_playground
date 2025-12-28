# -*- coding: utf-8 -*-
"""
Matplotlib 學習功能模組 - 第2部分：圓餅圖、直方圖、盒鬚圖
"""

# 繼續定義更多圖表類型

MATPLOTLIB_EFFECTS_PART2 = {
    # ===== 類別 4: 圓餅圖 =====
    'pie_basic': {
        'name': '基礎圓餅圖',
        'category': '圓餅圖',
        'description': '''圓餅圖用於顯示各部分占整體的比例，適合展示構成關係。

主要用途：
• 顯示市場佔有率
• 展示預算分配
• 表示時間分配
• 資源使用比例

基本語法：
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

重要參數：
• autopct: 顯示百分比格式 '%1.1f%%' 表示小數點後一位
• startangle: 起始角度（0-360度）
• explode: 突出顯示某些扇形
• shadow: 添加陰影效果
• colors: 自訂顏色列表

注意事項：
• 不建議超過7個類別
• 按大小排序更易閱讀
• 考慮使用圓環圖（Donut Chart）''',
        'params': [
            {'name': 'num_slices', 'label': '扇形數量', 'type': 'slider', 'min': 2, 'max': 8, 'step': 1, 'default': 5},
            {'name': 'explode_largest', 'label': '突出最大扇形', 'type': 'checkbox', 'default': False},
            {'name': 'show_shadow', 'label': '顯示陰影', 'type': 'checkbox', 'default': True},
            {'name': 'startangle', 'label': '起始角度', 'type': 'slider', 'min': 0, 'max': 360, 'step': 15, 'default': 90}
        ]
    },

    'pie_donut': {
        'name': '圓環圖',
        'category': '圓餅圖',
        'description': '''圓環圖是圓餅圖的變體，中間留有空洞，看起來更現代美觀。

優點：
• 視覺效果更好
• 可在中間顯示總計或關鍵數字
• 減少面積比較的誤差
• 適合多層數據展示

實現方法：
使用 plt.pie() 配合 wedgeprops 參數：
wedgeprops = {'width': 0.3}  # 0.3 是環的寬度

多層圓環圖：
可以繪製多個圓環來顯示層級數據
• 外圈顯示細分類
• 內圈顯示大分類''',
        'params': [
            {'name': 'num_slices', 'label': '扇形數量', 'type': 'slider', 'min': 3, 'max': 8, 'step': 1, 'default': 5},
            {'name': 'ring_width', 'label': '環寬度', 'type': 'slider', 'min': 0.2, 'max': 0.5, 'step': 0.05, 'default': 0.3},
            {'name': 'show_center_text', 'label': '中心文字', 'type': 'checkbox', 'default': True}
        ]
    },

    # ===== 類別 5: 直方圖 =====
    'hist_basic': {
        'name': '基礎直方圖',
        'category': '直方圖',
        'description': '''直方圖用於顯示數據的分布情況，將連續數據分組並統計頻率。

與長條圖的區別：
• 直方圖：顯示數值分布（連續數據）
• 長條圖：比較類別數量（離散數據）

基本語法：
plt.hist(data, bins=30, density=False, alpha=0.7, color='blue')

重要參數：
• bins: 分組數量或分組邊界
  - 整數：自動分成 n 組
  - 列表：自訂分組邊界
  - 'auto', 'sturges', 'scott': 自動計算最佳分組
• density: True 時顯示密度而非頻率
• cumulative: True 時顯示累積分布
• histtype: 'bar'(長條), 'step'(線條), 'stepfilled'(填充)

應用場景：
• 分析考試成績分布
• 觀察溫度、降雨量分布
• 質量控制（檢測產品規格）''',
        'params': [
            {'name': 'distribution', 'label': '分布類型', 'type': 'select',
             'options': [
                 {'value': 'normal', 'label': '常態分布'},
                 {'value': 'uniform', 'label': '均勻分布'},
                 {'value': 'exponential', 'label': '指數分布'},
                 {'value': 'bimodal', 'label': '雙峰分布'}
             ], 'default': 'normal'},
            {'name': 'num_samples', 'label': '樣本數', 'type': 'slider', 'min': 100, 'max': 2000, 'step': 100, 'default': 1000},
            {'name': 'bins', 'label': '分組數', 'type': 'slider', 'min': 10, 'max': 100, 'step': 5, 'default': 30},
            {'name': 'show_kde', 'label': '顯示密度曲線', 'type': 'checkbox', 'default': False}
        ]
    },

    'hist_compare': {
        'name': '對比直方圖',
        'category': '直方圖',
        'description': '''在同一圖表中顯示多個分布，用於比較不同組別的數據分布。

顯示方式：
1. 重疊 (Overlay)：
   使用透明度 (alpha) 讓兩組數據重疊顯示

2. 並排 (Side-by-side)：
   使用 histtype='bar' 和適當的位置偏移

3. 堆疊 (Stacked)：
   使用 stacked=True 參數

最佳實踐：
• 使用不同顏色區分組別
• 設置適當的透明度（建議 0.5-0.7）
• 添加清晰的圖例
• 使用相同的 bins 範圍以便比較

應用實例：
• 比較男女身高分布
• 對比處理前後的數據
• 比較不同班級的成績分布''',
        'params': [
            {'name': 'num_groups', 'label': '組別數', 'type': 'slider', 'min': 2, 'max': 4, 'step': 1, 'default': 2},
            {'name': 'overlap', 'label': '重疊顯示', 'type': 'checkbox', 'default': True},
            {'name': 'alpha', 'label': '透明度', 'type': 'slider', 'min': 0.3, 'max': 0.9, 'step': 0.1, 'default': 0.6}
        ]
    },

    # ===== 類別 6: 盒鬚圖 =====
    'box_basic': {
        'name': '基礎盒鬚圖',
        'category': '盒鬚圖',
        'description': '''盒鬚圖 (Box Plot) 用於顯示數據的分布特徵和離群值。

盒鬚圖的組成：
• 箱子 (Box)：表示四分位距 (IQR)
  - 下邊緣：第一四分位數 (Q1, 25%)
  - 中線：中位數 (Q2, 50%)
  - 上邊緣：第三四分位數 (Q3, 75%)
• 鬚 (Whiskers)：延伸到非離群值的最大/最小值
  - 通常是 1.5 × IQR
• 離群值：超出鬚範圍的點

基本語法：
plt.boxplot(data, labels=labels, showmeans=True, notch=True)

重要參數：
• vert: True(垂直) 或 False(水平)
• showmeans: 是否顯示平均值
• notch: 是否顯示凹槽（用於比較中位數）
• whis: 鬚的長度倍數（預設 1.5）

主要用途：
• 比較多組數據的分布
• 識別離群值
• 了解數據的對稱性和偏態''',
        'params': [
            {'name': 'num_boxes', 'label': '盒數', 'type': 'slider', 'min': 1, 'max': 6, 'step': 1, 'default': 3},
            {'name': 'orientation', 'label': '方向', 'type': 'select',
             'options': [
                 {'value': 'vertical', 'label': '垂直'},
                 {'value': 'horizontal', 'label': '水平'}
             ], 'default': 'vertical'},
            {'name': 'show_means', 'label': '顯示平均值', 'type': 'checkbox', 'default': True},
            {'name': 'show_notch', 'label': '顯示凹槽', 'type': 'checkbox', 'default': False}
        ]
    },

    'box_violin': {
        'name': '小提琴圖',
        'category': '盒鬚圖',
        'description': '''小提琴圖 (Violin Plot) 結合盒鬚圖和密度圖的優點。

特點：
• 形狀反映數據分布的密度
• 寬度表示該值的數據點密度
• 可以顯示盒鬚圖的統計信息
• 更能展現分布的細節（如雙峰）

與盒鬚圖比較：
• 盒鬚圖：簡潔，適合快速比較
• 小提琴圖：詳細，顯示完整分布形狀

語法：
parts = plt.violinplot(data, showmeans=True, showmedians=True)

參數：
• showmeans: 顯示平均值
• showmedians: 顯示中位數
• showextrema: 顯示極值
• widths: 小提琴的寬度

應用場景：
• 詳細的分布分析
• 比較多組數據的分布形狀
• 展示數據的多模態特性''',
        'params': [
            {'name': 'num_violins', 'label': '小提琴數', 'type': 'slider', 'min': 2, 'max': 5, 'step': 1, 'default': 3},
            {'name': 'show_box', 'label': '內嵌盒鬚圖', 'type': 'checkbox', 'default': True},
            {'name': 'bandwidth', 'label': '平滑程度', 'type': 'slider', 'min': 0.1, 'max': 1.0, 'step': 0.1, 'default': 0.3}
        ]
    },

    # ===== 類別 7: 熱力圖 =====
    'heatmap_basic': {
        'name': '基礎熱力圖',
        'category': '熱力圖',
        'description': '''熱力圖使用顏色深淺表示數值大小，適合展示矩陣數據。

主要用途：
• 相關性矩陣可視化
• 時間序列的模式分析
• 地理或空間數據
• 混淆矩陣（機器學習）

基本語法：
plt.imshow(data, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar()  # 添加色條

重要參數：
• cmap: 顏色映射
  - Sequential: 'viridis', 'plasma'（單向漸變）
  - Diverging: 'coolwarm', 'RdBu'（雙向對比）
  - 'hot', 'cool', 'gray' 等
• aspect: 圖形長寬比
• interpolation: 插值方法
  - 'nearest': 無插值（保持像素）
  - 'bilinear': 雙線性插值（平滑）

plt.pcolormesh() vs plt.imshow()：
• imshow: 更快，適合大數據
• pcolormesh: 支持不規則網格''',
        'params': [
            {'name': 'size', 'label': '矩陣大小', 'type': 'slider', 'min': 5, 'max': 20, 'step': 1, 'default': 10},
            {'name': 'colormap', 'label': '色彩映射', 'type': 'select',
             'options': [
                 {'value': 'viridis', 'label': 'Viridis（藍綠）'},
                 {'value': 'plasma', 'label': 'Plasma（紫橙）'},
                 {'value': 'coolwarm', 'label': 'Coolwarm（藍紅）'},
                 {'value': 'YlOrRd', 'label': 'YlOrRd（黃橙紅）'},
                 {'value': 'RdYlGn', 'label': 'RdYlGn（紅黃綠）'}
             ], 'default': 'viridis'},
            {'name': 'show_values', 'label': '顯示數值', 'type': 'checkbox', 'default': False},
            {'name': 'show_colorbar', 'label': '顯示色條', 'type': 'checkbox', 'default': True}
        ]
    },

    'heatmap_correlation': {
        'name': '相關性熱力圖',
        'category': '熱力圖',
        'description': '''相關性熱力圖是數據分析中最常用的工具之一，用於探索變數間的關係。

相關係數：
• 1: 完全正相關
• 0: 無相關
• -1: 完全負相關

視覺化技巧：
• 使用 diverging colormap（如 coolwarm, RdBu）
• 中心值設為 0
• 添加數值標註
• 考慮只顯示上三角或下三角（矩陣對稱）

應用：
• 特徵選擇（找出高度相關的特徵）
• 多重共線性檢測
• 變數關係探索

進階技巧：
• 使用 seaborn.heatmap() 更方便
• 配合 pandas.DataFrame.corr() 計算相關性
• 使用 mask 隱藏對角線或三角''',
        'params': [
            {'name': 'num_vars', 'label': '變數數量', 'type': 'slider', 'min': 3, 'max': 10, 'step': 1, 'default': 6},
            {'name': 'show_triangular', 'label': '只顯示下三角', 'type': 'checkbox', 'default': False}
        ]
    },

    # ===== 類別 8: 面積圖 =====
    'area_basic': {
        'name': '面積圖',
        'category': '面積圖',
        'description': '''面積圖是折線圖的延伸，填充線下方的區域，強調數量的變化。

主要用途：
• 顯示數量隨時間的變化和累積
• 比較多個數據系列的相對大小
• 展示部分與整體的關係

基本語法：
plt.fill_between(x, y1, y2, alpha=0.3, color='blue', label='區域')

重要參數：
• alpha: 透明度（建議 0.3-0.5）
• where: 條件性填充
• interpolate: 處理交叉點
• step: 階梯式填充

堆疊面積圖：
plt.stackplot(x, y1, y2, y3, labels=labels, alpha=0.7)

注意事項：
• 使用適當的透明度避免遮擋
• 堆疊時注意順序（大的在下）
• 不要堆疊太多層（≤5層）''',
        'params': [
            {'name': 'num_series', 'label': '數據系列數', 'type': 'slider', 'min': 1, 'max': 5, 'step': 1, 'default': 3},
            {'name': 'stacked', 'label': '堆疊顯示', 'type': 'checkbox', 'default': True},
            {'name': 'alpha', 'label': '透明度', 'type': 'slider', 'min': 0.3, 'max': 0.9, 'step': 0.1, 'default': 0.6}
        ]
    }
}
