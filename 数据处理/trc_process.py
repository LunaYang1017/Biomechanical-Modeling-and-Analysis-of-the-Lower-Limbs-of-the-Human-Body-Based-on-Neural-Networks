from tkinter import Frame

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

# 生成示例点位的名称（实际使用时应替换为您具体的点位名称）
point_names = ['Frame','SubFrame',
               'LASI_X', 'LASI_Y', 'LASI_Z',
               'RASI_X', 'RASI_Y', 'RASI_Z',
               'LPSI_X', 'LPSI_Y', 'LPSI_Z',
               'RPSI_X', 'RPSI_Y', 'RPSI_Z',
               'LTHI_X', 'LTHI_Y', 'LTHI_Z',
               'LKNE_X', 'LKNE_Y', 'LKNE_Z',
               'LANK_X', 'LANK_Y', 'LANK_Z',
               'LHEE_X', 'LHEE_Y', 'LHEE_Z',
               'LP1M_X', 'LP1M_Y', 'LP1M_Z',
               'LTOE_X', 'LTOE_Y', 'LTOE_Z',
               'LD5M_X', 'LD5M_Y', 'LD5M_Z',
               'LLCA_X', 'LLCA_Y', 'LLCA_Z',
               'RTHI_X', 'RTHI_Y', 'RTHI_Z',
               'RKNE_X', 'RKNE_Y', 'RKNE_Z',
               'RANK_X', 'RANK_Y', 'RANK_Z',
               'RHEE_X', 'RHEE_Y', 'RHEE_Z',
               'RP1M_X', 'RP1M_Y', 'RP1M_Z',
               'RTOE_X', 'RTOE_Y', 'RTOE_Z',
               'RD5M_X', 'RD5M_Y', 'RD5M_Z',
               'RLCA_X', 'RLCA_Y', 'RLCA_Z',
               ]


input_file = r"G:\动力学实验\22级系统动力学课程结课project说明\22级系统动力学课程结课项目题目\可选用数据_附件\标记点与足底力数据\run18_trc.csv"
output_file = r'G:\动力学实验\python\processed_run18_trc.csv'
# 读取数据
df = pd.read_csv(input_file, skiprows=5)

# # 1. 重命名列（使用更规范的命名方式）
# # 创建新的列名映射（20个点 × 3维坐标）
# columns_mapping = {}
#
# # 保留原始Frame列
# columns_mapping['Frame'] = 'Frame'
#
# # 为每个点位创建XYZ列
# for i, point in enumerate(point_names):
#     # 每组数据包含3列（X,Y,Z）
#     columns_mapping[f'Column_{3 * i + 1}'] = f'newproject_{point}_X'
#     columns_mapping[f'Column_{3 * i + 2}'] = f'newproject_{point}_Y'
#     columns_mapping[f'Column_{3 * i + 3}'] = f'newproject_{point}_Z'

# 2. 重命名列
df.columns = point_names

# 3. 处理缺失值（线性插值）
for col in df.columns:
    if col == 'Frame':  # 跳过Frame列
        continue

    # 创建连续编号索引（处理Frame中断情况）
    x = np.arange(len(df))
    y = df[col].values

    # 创建插值函数（仅非NaN点）
    mask = ~pd.isnull(y)
    if np.sum(mask) >= 2:  # 至少需要2个点才能插值
        interp_func = interp1d(x[mask], y[mask], kind='linear', fill_value='extrapolate')
        # 应用插值填充NaN
        df[col] = interp_func(x)
    else:
        # 不足2个点则用均值填充
        df[col] = df[col].fillna(df[col].mean())

# 4. 保存结果
if output_file is None:
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_processed{ext}"

df.to_csv(output_file, index=False)
print(f"处理完成! 结果已保存至: {output_file}")