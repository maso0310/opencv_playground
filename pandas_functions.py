"""
Pandas 資料處理功能模組
提供各種 Pandas 資料操作的教學與示範
"""

import pandas as pd
import numpy as np
import json
from io import StringIO

# Pandas 效果定義
PANDAS_EFFECTS = {
    # ===== DataFrame 建立 =====
    "create_from_dict": {
        "name": "從字典建立 DataFrame",
        "category": "DataFrame 建立",
        "description": "使用 Python 字典建立 DataFrame，字典的 key 為欄位名稱，value 為資料列表",
        "code_example": """import pandas as pd

data = {
    '姓名': ['小明', '小華', '小美'],
    '年齡': [25, 30, 28],
    '城市': ['台北', '台中', '高雄']
}
df = pd.DataFrame(data)
print(df)""",
        "params": [
            {"name": "columns", "type": "text", "default": "姓名,年齡,分數", "label": "欄位名稱（逗號分隔）"},
            {"name": "rows", "type": "number", "default": 5, "min": 1, "max": 20, "label": "資料筆數"}
        ]
    },
    "create_from_csv_string": {
        "name": "從 CSV 字串建立",
        "category": "DataFrame 建立",
        "description": "從 CSV 格式的字串建立 DataFrame，適合處理小型資料或測試",
        "code_example": """import pandas as pd
from io import StringIO

csv_data = '''姓名,年齡,城市
小明,25,台北
小華,30,台中'''

df = pd.read_csv(StringIO(csv_data))
print(df)""",
        "params": [
            {"name": "csv_content", "type": "textarea", "default": "產品,價格,數量\n蘋果,30,100\n香蕉,20,150\n橘子,25,80", "label": "CSV 內容"}
        ]
    },
    "create_random": {
        "name": "建立隨機資料",
        "category": "DataFrame 建立",
        "description": "建立包含隨機數值的 DataFrame，適合用於測試和練習",
        "code_example": """import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.random.randn(5, 3),
    columns=['A', 'B', 'C']
)
print(df)""",
        "params": [
            {"name": "rows", "type": "number", "default": 5, "min": 1, "max": 100, "label": "列數"},
            {"name": "cols", "type": "number", "default": 3, "min": 1, "max": 10, "label": "欄數"},
            {"name": "data_type", "type": "select", "default": "random", "options": ["random", "integers", "categories"], "label": "資料類型"}
        ]
    },
    "create_date_range": {
        "name": "建立日期序列資料",
        "category": "DataFrame 建立",
        "description": "建立包含日期索引的時間序列 DataFrame",
        "code_example": """import pandas as pd
import numpy as np

dates = pd.date_range('2024-01-01', periods=7, freq='D')
df = pd.DataFrame({
    '日期': dates,
    '溫度': np.random.randint(20, 35, 7),
    '濕度': np.random.randint(40, 80, 7)
})
print(df)""",
        "params": [
            {"name": "start_date", "type": "text", "default": "2024-01-01", "label": "起始日期"},
            {"name": "periods", "type": "number", "default": 7, "min": 1, "max": 365, "label": "天數"},
            {"name": "include_weather", "type": "checkbox", "default": True, "label": "包含天氣資料"}
        ]
    },

    # ===== 資料選取 =====
    "select_columns": {
        "name": "選取欄位",
        "category": "資料選取",
        "description": "從 DataFrame 中選取特定欄位",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['小明', '小華'],
    '年齡': [25, 30],
    '城市': ['台北', '台中']
})

# 選取單一欄位
print(df['姓名'])

# 選取多個欄位
print(df[['姓名', '年齡']])""",
        "params": [
            {"name": "columns_to_select", "type": "text", "default": "1,2", "label": "選取欄位索引（逗號分隔，從0開始）"}
        ]
    },
    "select_rows_by_index": {
        "name": "依索引選取列",
        "category": "資料選取",
        "description": "使用 iloc 依位置索引選取資料列",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['小明', '小華', '小美', '小強'],
    '分數': [85, 90, 78, 92]
})

# 選取前3列
print(df.iloc[:3])

# 選取第2到第4列
print(df.iloc[1:4])""",
        "params": [
            {"name": "start_idx", "type": "number", "default": 0, "min": 0, "max": 100, "label": "起始索引"},
            {"name": "end_idx", "type": "number", "default": 3, "min": 1, "max": 100, "label": "結束索引"}
        ]
    },
    "select_by_condition": {
        "name": "條件篩選",
        "category": "資料選取",
        "description": "根據條件篩選資料列",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['小明', '小華', '小美'],
    '分數': [85, 90, 78],
    '及格': [True, True, False]
})

# 篩選分數大於80的資料
result = df[df['分數'] > 80]
print(result)""",
        "params": [
            {"name": "column_idx", "type": "number", "default": 1, "min": 0, "max": 10, "label": "篩選欄位索引"},
            {"name": "operator", "type": "select", "default": ">", "options": [">", ">=", "<", "<=", "==", "!="], "label": "比較運算子"},
            {"name": "value", "type": "number", "default": 50, "label": "比較值"}
        ]
    },
    "loc_selection": {
        "name": "loc 標籤選取",
        "category": "資料選取",
        "description": "使用 loc 依標籤名稱選取資料",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['小明', '小華', '小美'],
    '數學': [85, 90, 78],
    '英文': [80, 85, 92]
}, index=['A', 'B', 'C'])

# 選取特定列和欄
print(df.loc['A':'B', ['姓名', '數學']])""",
        "params": [
            {"name": "row_start", "type": "number", "default": 0, "min": 0, "max": 100, "label": "列起始索引"},
            {"name": "row_end", "type": "number", "default": 2, "min": 0, "max": 100, "label": "列結束索引"},
            {"name": "col_indices", "type": "text", "default": "0,1", "label": "欄位索引（逗號分隔）"}
        ]
    },

    # ===== 資料過濾 =====
    "filter_by_value": {
        "name": "數值過濾",
        "category": "資料過濾",
        "description": "根據數值條件過濾資料",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '產品': ['A', 'B', 'C', 'D'],
    '價格': [100, 200, 150, 300],
    '庫存': [50, 30, 80, 20]
})

# 過濾價格在100-200之間的產品
result = df[(df['價格'] >= 100) & (df['價格'] <= 200)]
print(result)""",
        "params": [
            {"name": "column_idx", "type": "number", "default": 1, "min": 0, "max": 10, "label": "過濾欄位索引"},
            {"name": "min_value", "type": "number", "default": 0, "label": "最小值"},
            {"name": "max_value", "type": "number", "default": 100, "label": "最大值"}
        ]
    },
    "filter_by_string": {
        "name": "字串過濾",
        "category": "資料過濾",
        "description": "根據字串內容過濾資料",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['王小明', '李小華', '王大同', '陳小美'],
    '部門': ['業務', '研發', '業務', '人資']
})

# 過濾姓王的員工
result = df[df['姓名'].str.contains('王')]
print(result)""",
        "params": [
            {"name": "column_idx", "type": "number", "default": 0, "min": 0, "max": 10, "label": "過濾欄位索引"},
            {"name": "search_text", "type": "text", "default": "A", "label": "搜尋文字"},
            {"name": "match_type", "type": "select", "default": "contains", "options": ["contains", "startswith", "endswith", "exact"], "label": "比對方式"}
        ]
    },
    "filter_null": {
        "name": "過濾空值",
        "category": "資料過濾",
        "description": "過濾或保留包含空值的資料列",
        "code_example": """import pandas as pd
import numpy as np

df = pd.DataFrame({
    '姓名': ['小明', '小華', '小美'],
    '分數': [85, np.nan, 78],
    '等級': ['A', 'B', np.nan]
})

# 移除任何包含空值的列
clean_df = df.dropna()
print(clean_df)

# 只移除特定欄位有空值的列
clean_df2 = df.dropna(subset=['分數'])
print(clean_df2)""",
        "params": [
            {"name": "action", "type": "select", "default": "drop", "options": ["drop", "keep_null", "fill_zero", "fill_mean"], "label": "處理方式"},
            {"name": "column_idx", "type": "number", "default": -1, "min": -1, "max": 10, "label": "指定欄位（-1為全部）"}
        ]
    },
    "query_filter": {
        "name": "Query 查詢",
        "category": "資料過濾",
        "description": "使用 query 方法進行複雜條件查詢",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['小明', '小華', '小美', '小強'],
    '年齡': [25, 30, 28, 35],
    '薪資': [40000, 55000, 48000, 62000]
})

# 使用 query 進行複合條件查詢
result = df.query('年齡 > 25 and 薪資 > 45000')
print(result)""",
        "params": [
            {"name": "col1_idx", "type": "number", "default": 1, "min": 0, "max": 10, "label": "條件1欄位索引"},
            {"name": "op1", "type": "select", "default": ">", "options": [">", ">=", "<", "<=", "=="], "label": "條件1運算子"},
            {"name": "val1", "type": "number", "default": 50, "label": "條件1數值"},
            {"name": "logic", "type": "select", "default": "and", "options": ["and", "or"], "label": "邏輯運算"},
            {"name": "col2_idx", "type": "number", "default": 2, "min": 0, "max": 10, "label": "條件2欄位索引"},
            {"name": "op2", "type": "select", "default": ">", "options": [">", ">=", "<", "<=", "=="], "label": "條件2運算子"},
            {"name": "val2", "type": "number", "default": 30, "label": "條件2數值"}
        ]
    },

    # ===== 統計分析 =====
    "describe": {
        "name": "描述性統計",
        "category": "統計分析",
        "description": "計算 DataFrame 的描述性統計摘要",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '身高': [170, 165, 180, 175, 168],
    '體重': [65, 55, 80, 72, 60]
})

print(df.describe())""",
        "params": [
            {"name": "include_all", "type": "checkbox", "default": False, "label": "包含所有欄位類型"}
        ]
    },
    "mean_median_mode": {
        "name": "平均/中位/眾數",
        "category": "統計分析",
        "description": "計算資料的平均值、中位數和眾數",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '分數': [85, 90, 78, 92, 85, 88, 85]
})

print(f"平均值: {df['分數'].mean()}")
print(f"中位數: {df['分數'].median()}")
print(f"眾數: {df['分數'].mode()[0]}")""",
        "params": [
            {"name": "column_idx", "type": "number", "default": 0, "min": 0, "max": 10, "label": "計算欄位索引"},
            {"name": "stat_type", "type": "select", "default": "all", "options": ["all", "mean", "median", "mode"], "label": "統計類型"}
        ]
    },
    "correlation": {
        "name": "相關性分析",
        "category": "統計分析",
        "description": "計算欄位間的相關係數矩陣",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '身高': [170, 165, 180, 175, 168],
    '體重': [65, 55, 80, 72, 60],
    '年齡': [25, 30, 22, 28, 35]
})

print(df.corr())""",
        "params": [
            {"name": "method", "type": "select", "default": "pearson", "options": ["pearson", "spearman", "kendall"], "label": "相關係數方法"}
        ]
    },
    "value_counts": {
        "name": "值計數",
        "category": "統計分析",
        "description": "計算每個值出現的次數",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '等級': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'A']
})

print(df['等級'].value_counts())""",
        "params": [
            {"name": "column_idx", "type": "number", "default": 0, "min": 0, "max": 10, "label": "計算欄位索引"},
            {"name": "normalize", "type": "checkbox", "default": False, "label": "顯示百分比"},
            {"name": "sort", "type": "checkbox", "default": True, "label": "依數量排序"}
        ]
    },

    # ===== 群組操作 =====
    "groupby_basic": {
        "name": "基本群組",
        "category": "群組操作",
        "description": "依據指定欄位分組並計算統計值",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '部門': ['業務', '研發', '業務', '研發', '業務'],
    '姓名': ['小明', '小華', '小美', '小強', '小王'],
    '業績': [100, 80, 120, 90, 110]
})

result = df.groupby('部門')['業績'].sum()
print(result)""",
        "params": [
            {"name": "group_col", "type": "number", "default": 0, "min": 0, "max": 10, "label": "分組欄位索引"},
            {"name": "agg_col", "type": "number", "default": 1, "min": 0, "max": 10, "label": "聚合欄位索引"},
            {"name": "agg_func", "type": "select", "default": "sum", "options": ["sum", "mean", "count", "min", "max", "std"], "label": "聚合函數"}
        ]
    },
    "groupby_multiple": {
        "name": "多欄位群組",
        "category": "群組操作",
        "description": "依據多個欄位分組",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '年份': [2023, 2023, 2024, 2024],
    '季度': ['Q1', 'Q2', 'Q1', 'Q2'],
    '銷售額': [100, 120, 130, 150]
})

result = df.groupby(['年份', '季度'])['銷售額'].sum()
print(result)""",
        "params": [
            {"name": "group_cols", "type": "text", "default": "0,1", "label": "分組欄位索引（逗號分隔）"},
            {"name": "agg_col", "type": "number", "default": 2, "min": 0, "max": 10, "label": "聚合欄位索引"},
            {"name": "agg_func", "type": "select", "default": "sum", "options": ["sum", "mean", "count", "min", "max"], "label": "聚合函數"}
        ]
    },
    "groupby_agg": {
        "name": "多重聚合",
        "category": "群組操作",
        "description": "對分組資料套用多種聚合函數",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '部門': ['業務', '研發', '業務', '研發'],
    '薪資': [50000, 60000, 55000, 65000]
})

result = df.groupby('部門')['薪資'].agg(['mean', 'min', 'max'])
print(result)""",
        "params": [
            {"name": "group_col", "type": "number", "default": 0, "min": 0, "max": 10, "label": "分組欄位索引"},
            {"name": "agg_col", "type": "number", "default": 1, "min": 0, "max": 10, "label": "聚合欄位索引"}
        ]
    },
    "pivot_table": {
        "name": "樞紐分析表",
        "category": "群組操作",
        "description": "建立樞紐分析表（類似 Excel）",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '日期': ['週一', '週一', '週二', '週二'],
    '產品': ['A', 'B', 'A', 'B'],
    '銷量': [10, 15, 12, 18]
})

pivot = pd.pivot_table(df, values='銷量',
                       index='日期', columns='產品',
                       aggfunc='sum')
print(pivot)""",
        "params": [
            {"name": "index_col", "type": "number", "default": 0, "min": 0, "max": 10, "label": "列索引欄位"},
            {"name": "columns_col", "type": "number", "default": 1, "min": 0, "max": 10, "label": "欄索引欄位"},
            {"name": "values_col", "type": "number", "default": 2, "min": 0, "max": 10, "label": "數值欄位"},
            {"name": "agg_func", "type": "select", "default": "sum", "options": ["sum", "mean", "count", "min", "max"], "label": "聚合函數"}
        ]
    },

    # ===== 資料清理 =====
    "drop_duplicates": {
        "name": "移除重複值",
        "category": "資料清理",
        "description": "移除重複的資料列",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['小明', '小華', '小明', '小美'],
    '電話': ['0912', '0923', '0912', '0934']
})

clean_df = df.drop_duplicates()
print(clean_df)""",
        "params": [
            {"name": "subset_cols", "type": "text", "default": "", "label": "比對欄位索引（空為全部）"},
            {"name": "keep", "type": "select", "default": "first", "options": ["first", "last", "none"], "label": "保留方式"}
        ]
    },
    "fill_missing": {
        "name": "填補缺失值",
        "category": "資料清理",
        "description": "使用指定方法填補缺失值",
        "code_example": """import pandas as pd
import numpy as np

df = pd.DataFrame({
    '姓名': ['小明', '小華', '小美'],
    '分數': [85, np.nan, 78]
})

# 用平均值填補
df['分數'] = df['分數'].fillna(df['分數'].mean())
print(df)""",
        "params": [
            {"name": "method", "type": "select", "default": "mean", "options": ["mean", "median", "zero", "ffill", "bfill"], "label": "填補方法"},
            {"name": "column_idx", "type": "number", "default": -1, "min": -1, "max": 10, "label": "指定欄位（-1為全部數值欄）"}
        ]
    },
    "rename_columns": {
        "name": "重新命名欄位",
        "category": "資料清理",
        "description": "修改欄位名稱",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    'name': ['小明', '小華'],
    'age': [25, 30]
})

df = df.rename(columns={'name': '姓名', 'age': '年齡'})
print(df)""",
        "params": [
            {"name": "new_names", "type": "text", "default": "欄位A,欄位B,欄位C", "label": "新欄位名稱（逗號分隔）"}
        ]
    },
    "change_dtype": {
        "name": "轉換資料類型",
        "category": "資料清理",
        "description": "變更欄位的資料類型",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '數量': ['10', '20', '30'],
    '價格': ['100.5', '200.3', '150.8']
})

df['數量'] = df['數量'].astype(int)
df['價格'] = df['價格'].astype(float)
print(df.dtypes)""",
        "params": [
            {"name": "column_idx", "type": "number", "default": 0, "min": 0, "max": 10, "label": "欄位索引"},
            {"name": "new_type", "type": "select", "default": "int", "options": ["int", "float", "str", "bool", "datetime"], "label": "目標類型"}
        ]
    },

    # ===== 資料轉換 =====
    "sort_values": {
        "name": "排序",
        "category": "資料轉換",
        "description": "依據指定欄位排序資料",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['小明', '小華', '小美'],
    '分數': [85, 90, 78]
})

sorted_df = df.sort_values('分數', ascending=False)
print(sorted_df)""",
        "params": [
            {"name": "sort_col", "type": "number", "default": 0, "min": 0, "max": 10, "label": "排序欄位索引"},
            {"name": "ascending", "type": "checkbox", "default": True, "label": "升冪排序"}
        ]
    },
    "apply_function": {
        "name": "套用函數",
        "category": "資料轉換",
        "description": "對欄位套用自訂函數",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['小明', '小華'],
    '分數': [85, 90]
})

# 將分數轉換為等級
def get_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    else: return 'C'

df['等級'] = df['分數'].apply(get_grade)
print(df)""",
        "params": [
            {"name": "column_idx", "type": "number", "default": 0, "min": 0, "max": 10, "label": "套用欄位索引"},
            {"name": "operation", "type": "select", "default": "double", "options": ["double", "square", "sqrt", "abs", "round", "grade"], "label": "運算類型"}
        ]
    },
    "add_column": {
        "name": "新增欄位",
        "category": "資料轉換",
        "description": "基於現有欄位計算新增欄位",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '數量': [10, 20, 15],
    '單價': [100, 80, 120]
})

df['總價'] = df['數量'] * df['單價']
print(df)""",
        "params": [
            {"name": "col1_idx", "type": "number", "default": 0, "min": 0, "max": 10, "label": "欄位1索引"},
            {"name": "operator", "type": "select", "default": "+", "options": ["+", "-", "*", "/"], "label": "運算子"},
            {"name": "col2_idx", "type": "number", "default": 1, "min": 0, "max": 10, "label": "欄位2索引"},
            {"name": "new_col_name", "type": "text", "default": "計算結果", "label": "新欄位名稱"}
        ]
    },
    "melt_data": {
        "name": "寬轉長格式",
        "category": "資料轉換",
        "description": "將寬格式資料轉換為長格式（unpivot）",
        "code_example": """import pandas as pd

df = pd.DataFrame({
    '姓名': ['小明', '小華'],
    '國文': [85, 90],
    '英文': [80, 88],
    '數學': [92, 85]
})

melted = pd.melt(df, id_vars=['姓名'],
                 var_name='科目', value_name='分數')
print(melted)""",
        "params": [
            {"name": "id_cols", "type": "text", "default": "0", "label": "ID欄位索引（逗號分隔）"}
        ]
    },

    # ===== 合併操作 =====
    "merge_inner": {
        "name": "內部合併 (Inner Join)",
        "category": "合併操作",
        "description": "只保留兩個 DataFrame 都有的資料",
        "code_example": """import pandas as pd

df1 = pd.DataFrame({
    'ID': [1, 2, 3],
    '姓名': ['小明', '小華', '小美']
})

df2 = pd.DataFrame({
    'ID': [2, 3, 4],
    '分數': [90, 85, 88]
})

result = pd.merge(df1, df2, on='ID', how='inner')
print(result)""",
        "params": [
            {"name": "demo_type", "type": "select", "default": "students", "options": ["students", "products", "orders"], "label": "示範資料類型"}
        ]
    },
    "merge_left": {
        "name": "左合併 (Left Join)",
        "category": "合併操作",
        "description": "保留左邊 DataFrame 的所有資料",
        "code_example": """import pandas as pd

df1 = pd.DataFrame({
    'ID': [1, 2, 3],
    '姓名': ['小明', '小華', '小美']
})

df2 = pd.DataFrame({
    'ID': [2, 3, 4],
    '分數': [90, 85, 88]
})

result = pd.merge(df1, df2, on='ID', how='left')
print(result)""",
        "params": [
            {"name": "demo_type", "type": "select", "default": "students", "options": ["students", "products", "orders"], "label": "示範資料類型"}
        ]
    },
    "concat_rows": {
        "name": "垂直合併",
        "category": "合併操作",
        "description": "將多個 DataFrame 上下合併",
        "code_example": """import pandas as pd

df1 = pd.DataFrame({
    '姓名': ['小明', '小華'],
    '分數': [85, 90]
})

df2 = pd.DataFrame({
    '姓名': ['小美', '小強'],
    '分數': [78, 92]
})

result = pd.concat([df1, df2], ignore_index=True)
print(result)""",
        "params": [
            {"name": "ignore_index", "type": "checkbox", "default": True, "label": "重設索引"}
        ]
    },
    "concat_cols": {
        "name": "水平合併",
        "category": "合併操作",
        "description": "將多個 DataFrame 左右合併",
        "code_example": """import pandas as pd

df1 = pd.DataFrame({
    '姓名': ['小明', '小華'],
    '年齡': [25, 30]
})

df2 = pd.DataFrame({
    '城市': ['台北', '台中'],
    '職業': ['工程師', '設計師']
})

result = pd.concat([df1, df2], axis=1)
print(result)""",
        "params": []
    },
}

def get_all_pandas_effects():
    """取得所有 Pandas 效果的分類列表"""
    categories = {}
    for effect_id, effect in PANDAS_EFFECTS.items():
        category = effect.get("category", "其他")
        if category not in categories:
            categories[category] = []
        categories[category].append({
            "id": effect_id,
            "name": effect["name"],
            "description": effect["description"],
            "code_example": effect.get("code_example", ""),
            "params": effect.get("params", [])
        })
    return categories


def generate_sample_dataframe(rows=5, cols=3, data_type="mixed"):
    """生成範例 DataFrame"""
    np.random.seed(42)

    if data_type == "numeric":
        data = np.random.randn(rows, cols)
        columns = [f"數值{i+1}" for i in range(cols)]
        return pd.DataFrame(data, columns=columns)

    elif data_type == "categories":
        categories = ['A類', 'B類', 'C類']
        data = {
            f"類別{i+1}": np.random.choice(categories, rows)
            for i in range(cols)
        }
        return pd.DataFrame(data)

    else:  # mixed
        names = ['小明', '小華', '小美', '小強', '小王', '小李', '小陳', '小林']
        cities = ['台北', '台中', '高雄', '新竹', '台南']

        data = {
            '姓名': np.random.choice(names, rows),
            '年齡': np.random.randint(20, 50, rows),
            '城市': np.random.choice(cities, rows),
            '分數': np.random.randint(60, 100, rows),
            '薪資': np.random.randint(30000, 80000, rows)
        }
        return pd.DataFrame(data).iloc[:, :max(cols, 3)]


def df_to_html_table(df):
    """將 DataFrame 轉換為 HTML 表格"""
    return df.to_html(classes='pandas-table', index=True, border=0)


def process_pandas_operation(effect_name, params):
    """處理 Pandas 操作"""

    result = {
        "success": True,
        "output": "",
        "table_html": "",
        "code": "",
        "error": None
    }

    try:
        # DataFrame 建立
        if effect_name == "create_from_dict":
            columns = [c.strip() for c in params.get("columns", "姓名,年齡,分數").split(",")]
            rows = int(params.get("rows", 5))

            data = {}
            for i, col in enumerate(columns):
                if '年齡' in col or '分數' in col or '數' in col:
                    data[col] = np.random.randint(1, 100, rows).tolist()
                elif '薪資' in col or '金額' in col:
                    data[col] = np.random.randint(30000, 80000, rows).tolist()
                else:
                    names = ['小明', '小華', '小美', '小強', '小王', '小李']
                    data[col] = np.random.choice(names, rows).tolist()

            df = pd.DataFrame(data)
            result["table_html"] = df_to_html_table(df)
            result["output"] = f"建立了 {rows} 列 x {len(columns)} 欄的 DataFrame"
            result["code"] = f"df = pd.DataFrame({data})"

        elif effect_name == "create_from_csv_string":
            csv_content = params.get("csv_content", "A,B,C\n1,2,3")
            df = pd.read_csv(StringIO(csv_content))
            result["table_html"] = df_to_html_table(df)
            result["output"] = f"從 CSV 建立了 {len(df)} 列的 DataFrame"
            result["code"] = f"df = pd.read_csv(StringIO(csv_content))"

        elif effect_name == "create_random":
            rows = int(params.get("rows", 5))
            cols = int(params.get("cols", 3))
            data_type = params.get("data_type", "random")

            if data_type == "integers":
                df = pd.DataFrame(
                    np.random.randint(1, 100, (rows, cols)),
                    columns=[chr(65+i) for i in range(cols)]
                )
            elif data_type == "categories":
                cats = ['類別A', '類別B', '類別C']
                df = pd.DataFrame({
                    chr(65+i): np.random.choice(cats, rows)
                    for i in range(cols)
                })
            else:
                df = pd.DataFrame(
                    np.random.randn(rows, cols),
                    columns=[chr(65+i) for i in range(cols)]
                )

            result["table_html"] = df_to_html_table(df)
            result["output"] = f"建立了 {rows}x{cols} 的隨機 DataFrame"

        elif effect_name == "create_date_range":
            start_date = params.get("start_date", "2024-01-01")
            periods = int(params.get("periods", 7))
            include_weather = params.get("include_weather", True)

            dates = pd.date_range(start_date, periods=periods, freq='D')
            data = {'日期': dates}

            if include_weather:
                data['溫度(°C)'] = np.random.randint(18, 35, periods)
                data['濕度(%)'] = np.random.randint(40, 90, periods)
                data['降雨機率(%)'] = np.random.randint(0, 100, periods)

            df = pd.DataFrame(data)
            result["table_html"] = df_to_html_table(df)
            result["output"] = f"建立了 {periods} 天的時間序列資料"

        # 資料選取
        elif effect_name == "select_columns":
            df = generate_sample_dataframe(5, 5, "mixed")
            col_indices = [int(x.strip()) for x in params.get("columns_to_select", "1,2").split(",")]
            col_indices = [i for i in col_indices if i < len(df.columns)]

            selected = df.iloc[:, col_indices]
            result["table_html"] = df_to_html_table(selected)
            result["output"] = f"選取了欄位: {list(selected.columns)}"

        elif effect_name == "select_rows_by_index":
            df = generate_sample_dataframe(10, 4, "mixed")
            start = int(params.get("start_idx", 0))
            end = int(params.get("end_idx", 3))

            selected = df.iloc[start:end]
            result["table_html"] = df_to_html_table(selected)
            result["output"] = f"選取了第 {start} 到 {end-1} 列（共 {len(selected)} 列）"

        elif effect_name == "select_by_condition":
            df = generate_sample_dataframe(10, 5, "mixed")
            col_idx = int(params.get("column_idx", 1))
            op = params.get("operator", ">")
            val = float(params.get("value", 50))

            col_name = df.columns[min(col_idx, len(df.columns)-1)]

            # 確保是數值欄位
            if df[col_name].dtype in ['int64', 'float64']:
                if op == ">": mask = df[col_name] > val
                elif op == ">=": mask = df[col_name] >= val
                elif op == "<": mask = df[col_name] < val
                elif op == "<=": mask = df[col_name] <= val
                elif op == "==": mask = df[col_name] == val
                else: mask = df[col_name] != val

                selected = df[mask]
                result["table_html"] = df_to_html_table(selected)
                result["output"] = f"篩選 {col_name} {op} {val}，共 {len(selected)} 筆符合"
            else:
                result["output"] = f"欄位 {col_name} 不是數值型態，無法進行數值比較"
                result["table_html"] = df_to_html_table(df)

        elif effect_name == "loc_selection":
            df = generate_sample_dataframe(8, 5, "mixed")
            row_start = int(params.get("row_start", 0))
            row_end = int(params.get("row_end", 3))
            col_indices = [int(x.strip()) for x in params.get("col_indices", "0,1").split(",")]

            selected = df.iloc[row_start:row_end+1, col_indices]
            result["table_html"] = df_to_html_table(selected)
            result["output"] = f"選取了 {len(selected)} 列 x {len(selected.columns)} 欄"

        # 資料過濾
        elif effect_name == "filter_by_value":
            df = generate_sample_dataframe(10, 5, "mixed")
            col_idx = int(params.get("column_idx", 1))
            min_val = float(params.get("min_value", 0))
            max_val = float(params.get("max_value", 100))

            col_name = df.columns[min(col_idx, len(df.columns)-1)]

            if df[col_name].dtype in ['int64', 'float64']:
                mask = (df[col_name] >= min_val) & (df[col_name] <= max_val)
                filtered = df[mask]
                result["table_html"] = df_to_html_table(filtered)
                result["output"] = f"篩選 {min_val} <= {col_name} <= {max_val}，共 {len(filtered)} 筆"
            else:
                result["table_html"] = df_to_html_table(df)
                result["output"] = f"欄位 {col_name} 不是數值型態"

        elif effect_name == "filter_by_string":
            df = generate_sample_dataframe(10, 5, "mixed")
            col_idx = int(params.get("column_idx", 0))
            search_text = params.get("search_text", "小")
            match_type = params.get("match_type", "contains")

            col_name = df.columns[min(col_idx, len(df.columns)-1)]

            if match_type == "contains":
                mask = df[col_name].astype(str).str.contains(search_text, na=False)
            elif match_type == "startswith":
                mask = df[col_name].astype(str).str.startswith(search_text)
            elif match_type == "endswith":
                mask = df[col_name].astype(str).str.endswith(search_text)
            else:
                mask = df[col_name].astype(str) == search_text

            filtered = df[mask]
            result["table_html"] = df_to_html_table(filtered)
            result["output"] = f"在 {col_name} 中搜尋 '{search_text}'，找到 {len(filtered)} 筆"

        elif effect_name == "filter_null":
            # 建立含有空值的資料
            df = generate_sample_dataframe(8, 4, "mixed")
            df.iloc[2, 1] = np.nan
            df.iloc[5, 2] = np.nan

            action = params.get("action", "drop")
            col_idx = int(params.get("column_idx", -1))

            if action == "drop":
                if col_idx >= 0 and col_idx < len(df.columns):
                    filtered = df.dropna(subset=[df.columns[col_idx]])
                else:
                    filtered = df.dropna()
            elif action == "keep_null":
                filtered = df[df.isnull().any(axis=1)]
            elif action == "fill_zero":
                filtered = df.fillna(0)
            else:  # fill_mean
                filtered = df.copy()
                for col in filtered.select_dtypes(include=[np.number]).columns:
                    filtered[col] = filtered[col].fillna(filtered[col].mean())

            result["table_html"] = df_to_html_table(filtered)
            result["output"] = f"處理空值後剩餘 {len(filtered)} 筆資料"

        elif effect_name == "query_filter":
            df = generate_sample_dataframe(10, 5, "mixed")

            result["table_html"] = df_to_html_table(df)
            result["output"] = "Query 過濾示範（需要數值欄位）"

        # 統計分析
        elif effect_name == "describe":
            df = generate_sample_dataframe(20, 5, "mixed")
            include_all = params.get("include_all", False)

            if include_all:
                desc = df.describe(include='all')
            else:
                desc = df.describe()

            result["table_html"] = df_to_html_table(desc)
            result["output"] = "描述性統計摘要"

        elif effect_name == "mean_median_mode":
            df = generate_sample_dataframe(20, 5, "mixed")
            col_idx = int(params.get("column_idx", 1))
            stat_type = params.get("stat_type", "all")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col_name = numeric_cols[min(col_idx, len(numeric_cols)-1)]

                stats = {}
                if stat_type in ["all", "mean"]:
                    stats["平均值"] = df[col_name].mean()
                if stat_type in ["all", "median"]:
                    stats["中位數"] = df[col_name].median()
                if stat_type in ["all", "mode"]:
                    stats["眾數"] = df[col_name].mode().iloc[0] if len(df[col_name].mode()) > 0 else "N/A"

                stats_df = pd.DataFrame([stats], index=[col_name])
                result["table_html"] = df_to_html_table(stats_df)
                result["output"] = f"欄位 {col_name} 的統計值"
            else:
                result["output"] = "沒有數值欄位可計算"

        elif effect_name == "correlation":
            df = generate_sample_dataframe(20, 5, "mixed")
            method = params.get("method", "pearson")

            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr(method=method)

            result["table_html"] = df_to_html_table(corr.round(3))
            result["output"] = f"使用 {method} 方法計算相關係數"

        elif effect_name == "value_counts":
            df = generate_sample_dataframe(20, 5, "mixed")
            col_idx = int(params.get("column_idx", 0))
            normalize = params.get("normalize", False)
            sort = params.get("sort", True)

            col_name = df.columns[min(col_idx, len(df.columns)-1)]
            counts = df[col_name].value_counts(normalize=normalize, sort=sort)

            result["table_html"] = df_to_html_table(counts.to_frame())
            result["output"] = f"欄位 {col_name} 的值分布"

        # 群組操作
        elif effect_name == "groupby_basic":
            df = pd.DataFrame({
                '部門': np.random.choice(['業務', '研發', '行銷', '人資'], 20),
                '姓名': [f'員工{i}' for i in range(20)],
                '業績': np.random.randint(50, 150, 20),
                '薪資': np.random.randint(35000, 80000, 20)
            })

            group_col = int(params.get("group_col", 0))
            agg_col = int(params.get("agg_col", 2))
            agg_func = params.get("agg_func", "sum")

            group_name = df.columns[min(group_col, len(df.columns)-1)]
            agg_name = df.columns[min(agg_col, len(df.columns)-1)]

            if agg_func == "sum": grouped = df.groupby(group_name)[agg_name].sum()
            elif agg_func == "mean": grouped = df.groupby(group_name)[agg_name].mean()
            elif agg_func == "count": grouped = df.groupby(group_name)[agg_name].count()
            elif agg_func == "min": grouped = df.groupby(group_name)[agg_name].min()
            elif agg_func == "max": grouped = df.groupby(group_name)[agg_name].max()
            else: grouped = df.groupby(group_name)[agg_name].std()

            result["table_html"] = df_to_html_table(grouped.to_frame())
            result["output"] = f"依 {group_name} 分組，計算 {agg_name} 的 {agg_func}"

        elif effect_name == "groupby_multiple":
            df = pd.DataFrame({
                '年份': np.random.choice([2022, 2023, 2024], 20),
                '季度': np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], 20),
                '產品': np.random.choice(['A', 'B', 'C'], 20),
                '銷售額': np.random.randint(100, 500, 20)
            })

            group_cols = [int(x.strip()) for x in params.get("group_cols", "0,1").split(",")]
            agg_col = int(params.get("agg_col", 3))
            agg_func = params.get("agg_func", "sum")

            group_names = [df.columns[min(i, len(df.columns)-1)] for i in group_cols]
            agg_name = df.columns[min(agg_col, len(df.columns)-1)]

            grouped = df.groupby(group_names)[agg_name].agg(agg_func)
            result["table_html"] = df_to_html_table(grouped.to_frame())
            result["output"] = f"依 {group_names} 分組計算 {agg_func}"

        elif effect_name == "groupby_agg":
            df = pd.DataFrame({
                '部門': np.random.choice(['業務', '研發', '行銷'], 15),
                '薪資': np.random.randint(35000, 85000, 15)
            })

            group_col = int(params.get("group_col", 0))
            agg_col = int(params.get("agg_col", 1))

            group_name = df.columns[min(group_col, len(df.columns)-1)]
            agg_name = df.columns[min(agg_col, len(df.columns)-1)]

            grouped = df.groupby(group_name)[agg_name].agg(['mean', 'min', 'max', 'count'])
            result["table_html"] = df_to_html_table(grouped)
            result["output"] = f"依 {group_name} 分組的 {agg_name} 多重統計"

        elif effect_name == "pivot_table":
            df = pd.DataFrame({
                '日期': np.random.choice(['週一', '週二', '週三', '週四', '週五'], 20),
                '產品': np.random.choice(['產品A', '產品B', '產品C'], 20),
                '銷量': np.random.randint(10, 100, 20),
                '金額': np.random.randint(100, 1000, 20)
            })

            index_col = int(params.get("index_col", 0))
            columns_col = int(params.get("columns_col", 1))
            values_col = int(params.get("values_col", 2))
            agg_func = params.get("agg_func", "sum")

            pivot = pd.pivot_table(
                df,
                values=df.columns[values_col],
                index=df.columns[index_col],
                columns=df.columns[columns_col],
                aggfunc=agg_func,
                fill_value=0
            )

            result["table_html"] = df_to_html_table(pivot)
            result["output"] = f"樞紐分析表 - {agg_func}"

        # 資料清理
        elif effect_name == "drop_duplicates":
            df = pd.DataFrame({
                '姓名': ['小明', '小華', '小明', '小美', '小華', '小強'],
                '電話': ['0912', '0923', '0912', '0934', '0945', '0956'],
                '城市': ['台北', '台中', '台北', '高雄', '新竹', '台南']
            })

            subset_str = params.get("subset_cols", "")
            keep = params.get("keep", "first")

            if keep == "none":
                keep = False

            if subset_str:
                subset_cols = [int(x.strip()) for x in subset_str.split(",") if x.strip()]
                subset_names = [df.columns[i] for i in subset_cols if i < len(df.columns)]
                cleaned = df.drop_duplicates(subset=subset_names, keep=keep)
            else:
                cleaned = df.drop_duplicates(keep=keep)

            result["table_html"] = df_to_html_table(cleaned)
            result["output"] = f"移除重複後剩餘 {len(cleaned)} 筆（原 {len(df)} 筆）"

        elif effect_name == "fill_missing":
            df = pd.DataFrame({
                '姓名': ['小明', '小華', '小美', '小強', '小王'],
                '分數': [85.0, np.nan, 78.0, np.nan, 92.0],
                '等級': ['A', 'B', np.nan, 'C', 'A']
            })

            method = params.get("method", "mean")
            col_idx = int(params.get("column_idx", -1))

            filled = df.copy()

            if col_idx >= 0 and col_idx < len(df.columns):
                col_name = df.columns[col_idx]
                if method == "mean" and filled[col_name].dtype in ['float64', 'int64']:
                    filled[col_name] = filled[col_name].fillna(filled[col_name].mean())
                elif method == "median" and filled[col_name].dtype in ['float64', 'int64']:
                    filled[col_name] = filled[col_name].fillna(filled[col_name].median())
                elif method == "zero":
                    filled[col_name] = filled[col_name].fillna(0)
                elif method == "ffill":
                    filled[col_name] = filled[col_name].ffill()
                elif method == "bfill":
                    filled[col_name] = filled[col_name].bfill()
            else:
                for col in filled.columns:
                    if filled[col].dtype in ['float64', 'int64']:
                        if method == "mean":
                            filled[col] = filled[col].fillna(filled[col].mean())
                        elif method == "median":
                            filled[col] = filled[col].fillna(filled[col].median())
                        elif method == "zero":
                            filled[col] = filled[col].fillna(0)

            result["table_html"] = df_to_html_table(filled)
            result["output"] = f"使用 {method} 方法填補缺失值"

        elif effect_name == "rename_columns":
            df = generate_sample_dataframe(5, 4, "mixed")
            new_names = [n.strip() for n in params.get("new_names", "A,B,C,D").split(",")]

            rename_dict = {}
            for i, name in enumerate(new_names):
                if i < len(df.columns):
                    rename_dict[df.columns[i]] = name

            renamed = df.rename(columns=rename_dict)
            result["table_html"] = df_to_html_table(renamed)
            result["output"] = f"欄位已重新命名"

        elif effect_name == "change_dtype":
            df = pd.DataFrame({
                '數字文字': ['10', '20', '30', '40', '50'],
                '浮點文字': ['1.5', '2.5', '3.5', '4.5', '5.5'],
                '整數': [1, 2, 3, 4, 5]
            })

            col_idx = int(params.get("column_idx", 0))
            new_type = params.get("new_type", "int")

            if col_idx < len(df.columns):
                col_name = df.columns[col_idx]
                converted = df.copy()

                try:
                    if new_type == "int":
                        converted[col_name] = converted[col_name].astype(int)
                    elif new_type == "float":
                        converted[col_name] = converted[col_name].astype(float)
                    elif new_type == "str":
                        converted[col_name] = converted[col_name].astype(str)
                    elif new_type == "bool":
                        converted[col_name] = converted[col_name].astype(bool)

                    result["table_html"] = df_to_html_table(converted)
                    result["output"] = f"欄位 {col_name} 已轉換為 {new_type}，類型: {converted[col_name].dtype}"
                except Exception as e:
                    result["output"] = f"轉換失敗: {str(e)}"
                    result["table_html"] = df_to_html_table(df)
            else:
                result["table_html"] = df_to_html_table(df)

        # 資料轉換
        elif effect_name == "sort_values":
            df = generate_sample_dataframe(10, 5, "mixed")
            sort_col = int(params.get("sort_col", 1))
            ascending = params.get("ascending", True)

            col_name = df.columns[min(sort_col, len(df.columns)-1)]
            sorted_df = df.sort_values(col_name, ascending=ascending)

            result["table_html"] = df_to_html_table(sorted_df)
            direction = "升冪" if ascending else "降冪"
            result["output"] = f"依 {col_name} {direction}排序"

        elif effect_name == "apply_function":
            df = pd.DataFrame({
                '數值': [10, 25, 30, 45, 50, 65, 80, 95],
                '分數': [55, 65, 72, 78, 85, 90, 95, 100]
            })

            col_idx = int(params.get("column_idx", 0))
            operation = params.get("operation", "double")

            col_name = df.columns[min(col_idx, len(df.columns)-1)]
            applied = df.copy()

            if operation == "double":
                applied[f'{col_name}_加倍'] = applied[col_name] * 2
            elif operation == "square":
                applied[f'{col_name}_平方'] = applied[col_name] ** 2
            elif operation == "sqrt":
                applied[f'{col_name}_開根號'] = np.sqrt(applied[col_name])
            elif operation == "abs":
                applied[f'{col_name}_絕對值'] = np.abs(applied[col_name])
            elif operation == "round":
                applied[f'{col_name}_四捨五入'] = np.round(applied[col_name], 0)
            elif operation == "grade":
                def get_grade(x):
                    if x >= 90: return 'A'
                    elif x >= 80: return 'B'
                    elif x >= 70: return 'C'
                    elif x >= 60: return 'D'
                    return 'F'
                applied[f'{col_name}_等級'] = applied[col_name].apply(get_grade)

            result["table_html"] = df_to_html_table(applied)
            result["output"] = f"對 {col_name} 套用 {operation} 運算"

        elif effect_name == "add_column":
            df = pd.DataFrame({
                '數量': [10, 20, 15, 25, 30],
                '單價': [100, 80, 120, 90, 70],
                '折扣': [0.9, 0.85, 0.95, 0.8, 0.9]
            })

            col1 = int(params.get("col1_idx", 0))
            col2 = int(params.get("col2_idx", 1))
            op = params.get("operator", "*")
            new_name = params.get("new_col_name", "計算結果")

            col1_name = df.columns[min(col1, len(df.columns)-1)]
            col2_name = df.columns[min(col2, len(df.columns)-1)]

            if op == "+":
                df[new_name] = df[col1_name] + df[col2_name]
            elif op == "-":
                df[new_name] = df[col1_name] - df[col2_name]
            elif op == "*":
                df[new_name] = df[col1_name] * df[col2_name]
            elif op == "/":
                df[new_name] = df[col1_name] / df[col2_name]

            result["table_html"] = df_to_html_table(df)
            result["output"] = f"新增欄位: {new_name} = {col1_name} {op} {col2_name}"

        elif effect_name == "melt_data":
            df = pd.DataFrame({
                '姓名': ['小明', '小華', '小美'],
                '國文': [85, 90, 78],
                '英文': [80, 88, 92],
                '數學': [92, 75, 85]
            })

            id_cols_str = params.get("id_cols", "0")
            id_indices = [int(x.strip()) for x in id_cols_str.split(",") if x.strip()]
            id_vars = [df.columns[i] for i in id_indices if i < len(df.columns)]

            melted = pd.melt(df, id_vars=id_vars, var_name='科目', value_name='分數')

            result["table_html"] = df_to_html_table(melted)
            result["output"] = f"寬轉長格式完成，{len(df)} 列 -> {len(melted)} 列"

        # 合併操作
        elif effect_name == "merge_inner":
            demo_type = params.get("demo_type", "students")

            if demo_type == "students":
                df1 = pd.DataFrame({'ID': [1, 2, 3, 4], '姓名': ['小明', '小華', '小美', '小強']})
                df2 = pd.DataFrame({'ID': [2, 3, 4, 5], '分數': [90, 85, 88, 92]})
            elif demo_type == "products":
                df1 = pd.DataFrame({'產品ID': [101, 102, 103], '產品名': ['蘋果', '香蕉', '橘子']})
                df2 = pd.DataFrame({'產品ID': [102, 103, 104], '價格': [30, 25, 40]})
            else:
                df1 = pd.DataFrame({'訂單ID': [1, 2, 3], '客戶': ['A公司', 'B公司', 'C公司']})
                df2 = pd.DataFrame({'訂單ID': [2, 3, 4], '金額': [1000, 2000, 1500]})

            key_col = df1.columns[0]
            merged = pd.merge(df1, df2, on=key_col, how='inner')

            result["table_html"] = df_to_html_table(merged)
            result["output"] = f"Inner Join: {len(df1)} + {len(df2)} -> {len(merged)} 筆"

        elif effect_name == "merge_left":
            demo_type = params.get("demo_type", "students")

            if demo_type == "students":
                df1 = pd.DataFrame({'ID': [1, 2, 3, 4], '姓名': ['小明', '小華', '小美', '小強']})
                df2 = pd.DataFrame({'ID': [2, 3, 5], '分數': [90, 85, 92]})
            elif demo_type == "products":
                df1 = pd.DataFrame({'產品ID': [101, 102, 103], '產品名': ['蘋果', '香蕉', '橘子']})
                df2 = pd.DataFrame({'產品ID': [102, 104], '價格': [30, 40]})
            else:
                df1 = pd.DataFrame({'訂單ID': [1, 2, 3], '客戶': ['A公司', 'B公司', 'C公司']})
                df2 = pd.DataFrame({'訂單ID': [2, 4], '金額': [1000, 1500]})

            key_col = df1.columns[0]
            merged = pd.merge(df1, df2, on=key_col, how='left')

            result["table_html"] = df_to_html_table(merged)
            result["output"] = f"Left Join: 保留左表全部 {len(df1)} 筆"

        elif effect_name == "concat_rows":
            df1 = pd.DataFrame({
                '姓名': ['小明', '小華'],
                '年齡': [25, 30],
                '城市': ['台北', '台中']
            })
            df2 = pd.DataFrame({
                '姓名': ['小美', '小強'],
                '年齡': [28, 35],
                '城市': ['高雄', '新竹']
            })

            ignore_index = params.get("ignore_index", True)
            concatenated = pd.concat([df1, df2], ignore_index=ignore_index)

            result["table_html"] = df_to_html_table(concatenated)
            result["output"] = f"垂直合併: {len(df1)} + {len(df2)} = {len(concatenated)} 筆"

        elif effect_name == "concat_cols":
            df1 = pd.DataFrame({
                '姓名': ['小明', '小華', '小美'],
                '年齡': [25, 30, 28]
            })
            df2 = pd.DataFrame({
                '城市': ['台北', '台中', '高雄'],
                '職業': ['工程師', '設計師', '業務']
            })

            concatenated = pd.concat([df1, df2], axis=1)

            result["table_html"] = df_to_html_table(concatenated)
            result["output"] = f"水平合併: {len(df1.columns)} + {len(df2.columns)} = {len(concatenated.columns)} 欄"

        else:
            result["success"] = False
            result["error"] = f"未知的操作: {effect_name}"

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        import traceback
        result["output"] = traceback.format_exc()

    return result
