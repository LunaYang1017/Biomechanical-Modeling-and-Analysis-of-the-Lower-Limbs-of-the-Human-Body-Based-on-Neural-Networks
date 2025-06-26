import pandas as pd
import numpy as np

# 读取CSV文件，跳过前两行（文件头和单位行）
file_path = r'G:\动力学实验\22级系统动力学课程结课project说明\22级系统动力学课程结课项目题目\可选用数据_附件\标记点与足底力数据\run18_mot.csv'
df = pd.read_csv(file_path, skiprows=4)

# 检查列名，确保与数据匹配（根据实际文件调整）
# 原始数据列：Frame,Sub Frame,Fx,Fy,Fz,Mx,My,Mz,Cx,Cy,Cz
column_names = [
    'Frame', 'Sub Frame', 'Fx', 'Fy', 'Fz',
    'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz'
]
df.columns = column_names

# 1. 所有足底力乘上-1
df[['Fx', 'Fy', 'Fz']] = -df[['Fx', 'Fy', 'Fz']]

# 2. 单位统一：力分布中心(CP) mm转m，力矩(N·mm)转(N·m)
df[['Cx', 'Cy', 'Cz']] = df[['Cx', 'Cy', 'Cz']] / 1000  # mm -> m
df[['Mx', 'My', 'Mz']] = df[['Mx', 'My', 'Mz']] / 1000  # N·mm -> N·m

# 3. 跑台坐标系转换 (仅当数据为单足时应用)
if len(df.columns) == 11:  # 原始数据列数为11
    a = 1.5859  # 旋转角度(弧度)
    R_z = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a), np.cos(a), 0],
        [0, 0, 1]
    ])
    offset = np.array([724.5982, 385.3791, -3.6667]) / 1000  # 偏移量(m)

    # 对每个时间点应用转换
    for i in range(len(df)):
        cp_orig = df.loc[i, ['Cx', 'Cy', 'Cz']].values
        cp_new = R_z @ cp_orig + offset
        df.loc[i, ['Cx', 'Cy', 'Cz']] = cp_new

# 4. 移除力矩数据和Cz列
processed_df = df[['Frame', 'Sub Frame', 'Fx', 'Fy', 'Fz', 'Cx', 'Cy']]

# print(f"按Frame分组前数据点数: {len(processed_df)}")
# averaged_df = processed_df.groupby('Frame').agg({
#     'Fx': 'mean',
#     'Fy': 'mean',
#     'Fz': 'mean',
#     'Cx': 'mean',
#     'Cy': 'mean'
# }).reset_index()
# print(f"按Frame分组后数据点数: {len(averaged_df)}")

# 保存处理后的数据
output_path = r'G:\动力学实验\python\arv_processed_run18_mot.csv'
processed_df.to_csv(output_path, index=False)

print(f"数据处理完成！结果已保存至: {output_path}")


print("输出列包括: Frame, Sub Frame, Fx, Fy, Fz, Cx, Cy")

# 实际使用时应替换为：
data = pd.read_csv(r'G:\动力学实验\python\arv_processed_run18_mot.csv')

# 按Frame分组计算平均值
agg_dict = {col: 'mean' for col in data.columns if col not in ['Frame', 'Sub Frame']}
result = data.groupby('Frame', as_index=False).agg(agg_dict)

# 保存处理后的数据
output_path = r'G:\动力学实验\python\arv_processed_run18_mot.csv'
result.to_csv(output_path, index=False)