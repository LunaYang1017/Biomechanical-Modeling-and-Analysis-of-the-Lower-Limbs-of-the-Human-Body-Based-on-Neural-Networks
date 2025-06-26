
# 动力学响应分析与训练项目

本项目包含动力学仿真、数据处理和神经网络训练三部分工作流程。

## 目录结构
```
├── model_simulate.py # 动力学仿真与瞬态响应脚本
├── data_processing/ # 数据处理模块
│ ├── mot_process.py # 运动数据处理器
│ ├── trc_process.py # 轨迹数据处理器
│ └── train_run18.csv # 生成的训练数据
├── train/ # 神经网络训练模块
│ ├── transform_train.py # 变换模型训练
│ └── train_FZ2/ # FZ2 模型专用训练
│ └── train_FZsp.py # FZSP 模型训练
└── data/ # 原始数据集 (示例)
├── run18_mot.csv
└── run18_trc.csv
```

## 运行流程

### 1. 仿真与瞬态响应
运行动力学仿真生成基本响应数据：
```
python model_simulate.py
```
### 2. 数据处理 (基于 run18 数据集)
分步处理原始数据并生成训练集：

​​处理运动数据​​：
```
python mot_process.py 
```
​​处理轨迹数据​​：
```
python trc_process.py 
```
​​合并数据生成训练集​​：
```
手动合并三个对齐的CSV文件 
```
输出文件
```
 train_run18.csv
 ```
### 3. 训练神经网络
#### 初始模型训练
修改 train/transform_train.py 中的数据集为train_run18.csv路径后运行：
```
python train/transform_train.py
```
#### 针对Z方向优化的模型训练
修改 train_FZ2/train_FZsp.py 中的数据集路径
启动训练：
```
python train_FZ2/train_FZsp.py
```
