import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import os

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Run18Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Run18Model, self).__init__()
        self.model = nn.Sequential(
            # 输入层
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 隐藏层1
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 隐藏层2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # 输出层（无激活函数）
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.model(x)


def load_and_preprocess_data(csv_path, input_cols, output_cols):
    """加载和预处理数据"""
    df = pd.read_csv(csv_path)

    # 确保所有列都存在
    missing_cols = [col for col in input_cols + output_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下列在数据集中缺失: {', '.join(missing_cols)}")

    # 提取输入输出数据
    X = df[input_cols].values
    y = df[output_cols].values

    # 数据标准化
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    return X_scaled, y_scaled, x_scaler, y_scaler


def train_model(model, train_loader, val_loader, epochs=400, lr=0.001):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

        # 计算平均损失
        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(val_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'train_run18_model_best.pth')

        # 更新学习率
        scheduler.step(epoch_val_loss)

        # 打印进度
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {epoch_train_loss:.4f}, '
                  f'Val Loss: {epoch_val_loss:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader, y_scaler):
    """评估模型性能"""
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # 保存结果
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    # 合并所有batch的结果
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)

    # 逆标准化
    targets_orig = y_scaler.inverse_transform(all_targets)
    predictions_orig = y_scaler.inverse_transform(all_predictions)

    # 计算性能指标
    mae = mean_absolute_error(targets_orig, predictions_orig)
    r2 = r2_score(targets_orig, predictions_orig)

    print(f'\nFinal Evaluation:')
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'R² Score: {r2:.4f}')

    # 输出各维度的MAE
    output_names = ['Fx', 'Fy', 'Fz', 'Cx', 'Cy']
    for i, name in enumerate(output_names):
        dim_mae = mean_absolute_error(targets_orig[:, i], predictions_orig[:, i])
        print(f'MAE for {name}: {dim_mae:.4f}')

    return targets_orig, predictions_orig


def plot_results(train_loss, val_loss, targets, predictions, sample_index=0):
    """可视化结果"""
    plt.figure(figsize=(18, 12))

    # 1. 绘制训练和验证损失
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. 绘制特定样本的预测结果
    plt.subplot(2, 2, 2)
    features = ['Fx', 'Fy', 'Fz', 'Cx', 'Cy']
    x = np.arange(len(features))
    width = 0.35

    plt.bar(x - width / 2, targets[sample_index], width, label='True Value')
    plt.bar(x + width / 2, predictions[sample_index], width, label='Predicted Value')

    plt.xlabel('Biomechanical Parameters')
    plt.ylabel('Value')
    plt.title(f'Prediction vs True Value (Sample {sample_index})')
    plt.xticks(x, features)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. 绘制全部样本的真实值vs预测值散点图
    plt.subplot(2, 2, 3)
    plt.scatter(targets.ravel(), predictions.ravel(), alpha=0.3, s=10)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predicted vs True Values')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. 绘制各维度的误差分布
    plt.subplot(2, 2, 4)
    errors = predictions - targets
    abs_errors = np.abs(errors)

    plt.boxplot(abs_errors,  tick_labels=features, showfliers=False)
    plt.ylabel('Absolute Error')
    plt.title('Error Distribution by Parameter')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('train_run18_model_results.png', dpi=300)
    plt.show()

    # 额外绘制每个参数的预测结果对比
    plt.figure(figsize=(15, 10))
    n_subplots = len(features)
    for i, feature in enumerate(features):
        plt.subplot(n_subplots, 1, i + 1)
        true_vals = targets[:, i]
        pred_vals = predictions[:, i]

        # 仅绘制前200个样本更清晰
        idx = np.arange(min(200, len(true_vals)))
        plt.plot(idx, true_vals[:200], 'b-', label='True', alpha=0.8)
        plt.plot(idx, pred_vals[:200], 'r--', label='Predicted', alpha=0.8)

        plt.ylabel(feature)
        if i == n_subplots - 1:
            plt.xlabel('Sample Index')
        if i == 0:
            plt.legend()
            plt.title('Parameter Prediction Over Samples')
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('parameter_comparisons.png', dpi=300)
    plt.show()


def main():
    # 配置参数 - 使用实际标记点名称
    input_cols = [
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
        '1','2','3','4','5','6'
    ]

    output_cols = ['Fx', 'Fy', 'Fz', 'Cx', 'Cy']

    csv_path = r'G:\动力学实验\python\data\train\train_run18.csv'  # 确保此路径正确

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件未找到: {csv_path}")

    # 1. 加载数据
    print("加载和预处理数据...")
    try:
        X, y, x_scaler, y_scaler = load_and_preprocess_data(csv_path, input_cols, output_cols)
        print(f"成功加载数据，样本数: {X.shape[0]}, 输入特征数: {X.shape[1]}, 输出维度: {y.shape[1]}")
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        return

    # 2. 创建PyTorch数据集
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    # 3. 划分训练集、验证集、测试集 (70%-15%-15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 4. 创建数据加载器
    batch_size = 64  # 增加批量大小以提高训练稳定性
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 5. 初始化模型
    model = Run18Model(input_size=len(input_cols), output_size=len(output_cols)).to(device)

    # 打印模型概况
    print(f"\n模型结构:\n{model}\n")
    print(f"将在 {device} 上训练")
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}, 测试样本数: {len(test_dataset)}")

    # 6. 训练模型
    print("\n开始模型训练...")
    try:
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, epochs=400, lr=0.0005
        )
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        return

    # 7. 加载最佳模型进行评估
    try:
        model.load_state_dict(torch.load('train_run18_model_best.pth'))
        print("加载最佳模型进行最终评估...")
    except:
        print("使用最终模型进行评估...")

    # 8. 评估模型
    print("\n评估模型性能...")
    targets, predictions = evaluate_model(model, test_loader, y_scaler)

    # 9. 可视化结果
    print("\n生成可视化图表...")
    plot_results(train_losses, val_losses, targets, predictions)

    # 10. 保存最终模型
    torch.save(model.state_dict(), 'train_run18_model_final.pth')
    print("\n最终模型已保存为 train_run18_model_final.pth")


if __name__ == "__main__":
    main()