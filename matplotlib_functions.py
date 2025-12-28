# -*- coding: utf-8 -*-
"""
Matplotlib 學習功能模組
整合所有 Matplotlib 圖表定義
"""

# 導入各部分定義
from matplotlib_functions_part1 import MATPLOTLIB_EFFECTS_PART1
from matplotlib_functions_part2 import MATPLOTLIB_EFFECTS_PART2

# 整合所有效果
MATPLOTLIB_EFFECTS = {}
MATPLOTLIB_EFFECTS.update(MATPLOTLIB_EFFECTS_PART1)
MATPLOTLIB_EFFECTS.update(MATPLOTLIB_EFFECTS_PART2)


def get_all_matplotlib_effects():
    """取得所有 Matplotlib 效果的定義"""
    return MATPLOTLIB_EFFECTS


# 導入處理函數
from matplotlib_processors import process_matplotlib_plot


def process_matplotlib_operation(effect, params):
    """
    處理 Matplotlib 操作

    參數:
        effect: 效果類型
        params: 參數字典

    返回:
        (image_base64, code): 圖片 base64 字串和對應程式碼
    """
    try:
        image_base64, code = process_matplotlib_plot(effect, params)

        if image_base64 is None:
            return None, "# 尚未實現此圖表"

        return {
            'success': True,
            'image': image_base64,
            'code': code,
            'effect_info': MATPLOTLIB_EFFECTS.get(effect, {})
        }
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
