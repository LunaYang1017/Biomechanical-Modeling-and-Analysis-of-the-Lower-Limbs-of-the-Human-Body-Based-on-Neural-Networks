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
import joblib
import torch.nn.functional as F

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 混合精度训练启用
use_amp = True


class CombinedLoss(nn.Module):
    """组合损失函数：Fz使用加权MAE，其他输出使用MSE，添加时间平滑性约束"""

    def __init__(self, fz_weight=5.0, temporal_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.fz_weight = fz_weight
        self.temporal_weight = temporal_weight

    def forward(self, output, target):
        # Fz的MAE损失（更高权重）
        fz_loss = F.l1_loss(output[:, 2], target[:, 2]) * self.fz_weight

        # 其他输出的MSE损失
        other_indices = [i for i in range(target.shape[1]) if i != 2]
        other_loss = F.mse_loss(output[:, other_indices], target[:, other_indices])

        # 时间平滑性约束（减小帧间跳跃）
        if output.shape[0] > 1:  # 确保批次有多个样本
            temporal_loss = torch.mean(torch.abs(output[1:] - output[:-1]))
        else:
            temporal_loss = 0.0

        return fz_loss + other_loss + self.temporal_weight * temporal_loss


class EnhancedFzModel(nn.Module):
    """优化版模型架构：双路径网络处理Fz特殊需求"""

    def __init__(self, input_size, output_size):
        super(EnhancedFzModel, self).__init__()

        # 共享特征提取层
        self.shared_features = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )

        # Fz专项路径
        self.fz_path = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),  # Swish激活函数

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            nn.Linear(64, 1)
        )

        # 其他输出路径
        self.other_path = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Linear(128, output_size - 1)  # 减去Fz的一个输出
        )

    def forward(self, x):
        features = self.shared_features(x)

        # Fz输出
        fz_output = self.fz_path(features)

        # 其他输出
        other_outputs = self.other_path(features)

        # 组合输出：保持原始输出顺序[Fx, Fy, Fz, Cx, Cy]
        full_output = torch.cat([
            other_outputs[:, :2],  # Fx, Fy
            fz_output,  # Fz
            other_outputs[:, 2:]  # Cx, Cy
        ], dim=1)

        return full_output


def enhanced_data_preprocessing(csv_path, input_cols, output_cols):
    """增强版数据预处理：异常值处理、特征工程、Fz单独标准化"""
    # 加载数据
    df = pd.read_csv(csv_path)

    # 1. 异常值处理：过滤Fz明显异常的数据
    fz_mask = (df['Fz'] > 50) & (df['Fz'] < 2000)
    df = df[fz_mask]
    print(f"异常值过滤后样本数: {len(df)}")

    # 2. 特征工程：添加骨盆相关动态特征
    # 骨盆中心点计算
    df['Pelvis_Center_X'] = (df['LASI_X'] + df['RASI_X']) / 2
    df['Pelvis_Center_Z'] = (df['LASI_Z'] + df['RASI_Z']) / 2
    # 骨盆宽度
    df['Pelvis_Width'] = df['RASI_X'] - df['LASI_X']
    # 矢状面位移（踝关节到骨盆）
    df['Sagittal_Displacement'] = df['LASI_Z'] - df['LHEE_Z']

    # 更新输入列（添加新特征）
    enhanced_input_cols = input_cols + [
        'Pelvis_Center_X', 'Pelvis_Center_Z',
        'Pelvis_Width', 'Sagittal_Displacement'
    ]

    # 数据标准化
    X = df[enhanced_input_cols].values
    y = df[output_cols].values

    # 3. Fz单独标准化
    fz_idx = output_cols.index('Fz')
    print(f"Fz所在输出位置: {fz_idx}")

    fz_scaler = StandardScaler()
    y_fz_scaled = fz_scaler.fit_transform(y[:, fz_idx].reshape(-1, 1))

    # 其他输出标准化
    other_output_scaler = StandardScaler()
    other_outputs = np.delete(y, fz_idx, axis=1)
    y_other_scaled = other_output_scaler.fit_transform(other_outputs)

    # 合并标准化后的输出
    y_scaled = np.zeros_like(y)
    y_scaled[:, fz_idx] = y_fz_scaled.flatten()
    other_idx = [i for i in range(y.shape[1]) if i != fz_idx]
    for idx, col_idx in enumerate(other_idx):
        y_scaled[:, col_idx] = y_other_scaled[:, idx]

    # 输入标准化
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)

    return X_scaled, y_scaled, x_scaler, other_output_scaler, fz_scaler, enhanced_input_cols


def train_enhanced_model(model, train_loader, val_loader, loss_fn, epochs=600, init_lr=0.0005):
    """增强版训练函数：整合所有先进训练策略"""
    # 使用AdamW优化器（带权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-4)

    # OneCycleLR学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=init_lr * 10,  # 最大学习率是初始值的10倍
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    # 梯度裁剪值
    clip_value = 1.0

    # 混合精度训练（如果可用）
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 早停机制配置
    best_val_loss = float('inf')
    patience = 100  # 连续无改善的epoch数
    patience_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # 训练阶段（混合精度）
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # 混合精度作用域
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            # 反向传播与优化（考虑混合精度）
            scaler.scale(loss).backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            scaler.step(optimizer)
            scaler.update()

            # 更新学习率
            scheduler.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 早停和模型保存逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_enhanced_model.pth')
            patience_counter = 0
            print(f"保存最佳模型，验证损失: {val_loss:.6f}")
        else:
            patience_counter += 1

        # 打印训练信息（每10个epoch）
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.6f} | '
                  f'Val Loss: {val_loss:.6f} | LR: {current_lr:.8f} | Patience: {patience_counter}/{patience}')

        # 早停检查
        if patience_counter >= patience:
            print(f'\n早停触发，第 {epoch + 1} 轮（连续 {patience} 轮验证损失无改善）')
            print(f'最佳验证损失: {best_val_loss:.6f}')
            break

    return train_losses, val_losses, best_val_loss


def evaluate_enhanced_model(model, test_loader, fz_scaler, other_scaler, output_cols):
    """增强版模型评估函数：考虑双标准化器"""
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)

            # 混合精度预测（如果启用）
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)

            # 保存结果
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    # 合并所有batch的结果
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)

    # 逆向标准化（特殊处理Fz）
    # 1. 分离Fz和其他数据
    fz_idx = output_cols.index('Fz')
    fz_targets = all_targets[:, fz_idx]
    fz_preds = all_predictions[:, fz_idx]

    other_targets = np.delete(all_targets, fz_idx, axis=1)
    other_preds = np.delete(all_predictions, fz_idx, axis=1)

    # 2. 分别逆向标准化
    fz_targets_orig = fz_scaler.inverse_transform(fz_targets.reshape(-1, 1))
    fz_preds_orig = fz_scaler.inverse_transform(fz_preds.reshape(-1, 1))

    other_targets_orig = other_scaler.inverse_transform(other_targets)
    other_preds_orig = other_scaler.inverse_transform(other_preds)

    # 3. 重新组合原始数据
    targets_orig = np.zeros((len(all_targets), len(output_cols)))
    predictions_orig = np.zeros((len(all_targets), len(output_cols)))

    targets_orig[:, fz_idx] = fz_targets_orig.flatten()
    predictions_orig[:, fz_idx] = fz_preds_orig.flatten()

    other_idx = [i for i in range(len(output_cols)) if i != fz_idx]
    for i, idx in enumerate(other_idx):
        targets_orig[:, idx] = other_targets_orig[:, i]
        predictions_orig[:, idx] = other_preds_orig[:, i]

    # 计算性能指标
    mae = mean_absolute_error(targets_orig, predictions_orig)
    r2 = r2_score(targets_orig, predictions_orig)

    print(f'\n最终评估结果:')
    print(f'全局平均绝对误差(MAE): {mae:.4f}')
    print(f'决定系数(R²): {r2:.4f}')

    # 输出各维度的MAE
    for i, name in enumerate(output_cols):
        dim_mae = mean_absolute_error(targets_orig[:, i], predictions_orig[:, i])
        print(f'{name} 维度误差(MAE): {dim_mae:.4f}')

    return targets_orig, predictions_orig


def plot_enhanced_results(train_loss, val_loss, targets, predictions,  output_cols, best_val_loss,sample_index=0 ,):
    """增强版结果可视化函数"""
    # plt.figure(figsize=(18, 15))
    #
    # # 1. 训练损失曲线
    # plt.subplot(3, 2, 1)
    # plt.plot(train_loss, label='Training Loss', linewidth=2)
    # plt.plot(val_loss, label='Validation Loss', linewidth=2)
    # plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val Loss ({best_val_loss:.4f})')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.yscale('log')  # 对数坐标更清晰
    #
    # # 2. Fz预测对比（随机样本）
    # sample_idx = np.random.randint(0, len(targets))
    # fz_idx = output_cols.index('Fz')
    #
    # plt.subplot(3, 2, 2)
    # plt.plot(targets[sample_idx - 50:sample_idx + 50, fz_idx], 'b-', label='真实值')
    # plt.plot(predictions[sample_idx - 50:sample_idx + 50, fz_idx], 'r--', label='预测值')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Fz (N)')
    # plt.title(f'Fz Prediction Comparison  (Samples{sample_idx - 50}到{sample_idx + 50})')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.5)
    #
    # # 3. 所有输出参数的误差分布
    # plt.subplot(3, 2, 3)
    # errors = predictions - targets
    # abs_errors = np.abs(errors)
    #
    # plt.boxplot(abs_errors, labels=output_cols, showfliers=False)
    # plt.ylabel('Absolute Error')
    # plt.title('Absolute Error Distribution by Parameter')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    #
    # # 4. Fz预测-真实值散点图
    # plt.subplot(3, 2, 4)
    # plt.scatter(targets[:, fz_idx], predictions[:, fz_idx], alpha=0.3, s=15)
    # plt.plot([targets[:, fz_idx].min(), targets[:, fz_idx].max()],
    #          [targets[:, fz_idx].min(), targets[:, fz_idx].max()],
    #          'r--', linewidth=2)
    # plt.xlabel('True Fz (N)')
    # plt.ylabel('Predicted Fz (N)')
    # plt.title(f'Fz Prediction Accuracy  (MAE = {mean_absolute_error(targets[:, fz_idx], predictions[:, fz_idx]):.2f}N)')
    # plt.grid(True, linestyle='--', alpha=0.5)
    #
    # # 5. 参数预测趋势对比（随机5个样本）
    # sample_indices = np.random.choice(len(targets), 5, replace=False)
    #
    # plt.subplot(3, 2, 5)
    # for i, idx in enumerate(sample_indices):
    #     # 只绘制Fz值
    #     plt.plot([i - 0.2, i + 0.2], [targets[idx, fz_idx], targets[idx, fz_idx]], 'b-', linewidth=3)
    #     plt.plot([i - 0.2, i + 0.2], [predictions[idx, fz_idx], predictions[idx, fz_idx]], 'r--', linewidth=3)
    #
    # plt.xticks(range(5), [f'样本 {idx}' for idx in sample_indices])
    # plt.ylabel('Fz (N)')
    # plt.title('Fz Prediction Comparison for Random Samples')
    # plt.legend(['True Value', 'Predicted Value'], loc='upper right')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    #
    # # 6. 各参数预测准确性对比
    # plt.subplot(3, 2, 6)
    # maes = [mean_absolute_error(targets[:, i], predictions[:, i]) for i in range(len(output_cols))]
    # plt.bar(output_cols, maes, color=['blue' if name != 'Fz' else 'red' for name in output_cols])
    # plt.ylabel('MAE')
    # plt.title('Mean Absolute Error by Output Parameter')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    #
    # # 添加全局标题
    # plt.suptitle('Model Performance Analysis', fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    #
    # plt.savefig('enhanced_model_performance.png', dpi=300)
    # plt.show()
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

    # for i, idx in enumerate(sample_indices):
    #     # 只绘制Fz值
    #     plt.plot([i - 0.2, i + 0.2], [targets[idx, fz_idx], targets[idx, fz_idx]], 'b-', linewidth=3)
    #     plt.plot([i - 0.2, i + 0.2], [predictions[idx, fz_idx], predictions[idx, fz_idx]], 'r--', linewidth=3)
    #
    # plt.xticks(range(5), [f'样本 {idx}' for idx in sample_indices])
    # plt.ylabel('Fz (N)')
    # plt.title('Fz Prediction Comparison for Random Samples')
    # plt.legend(['True Value', 'Predicted Value'], loc='upper right')
    # plt.grid(True, axis='y', linestyle='--', alpha=0.5)
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

    plt.boxplot(abs_errors, tick_labels=features, showfliers=False)
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
    # 初始化配置
    print("=" * 50)
    print("Fz优化模型训练 - 增强版")
    print("=" * 50)

    # 原始输入列定义
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
        '1', '2', '3', '4', '5', '6'
    ]

    output_cols = ['Fx', 'Fy', 'Fz', 'Cx', 'Cy']

    # 数据文件路径
    csv_path = r'G:\动力学实验\python\data\train\train_run18.csv'

    # 1. 增强数据预处理
    print("\n开始数据预处理...")
    try:
        X_scaled, y_scaled, x_scaler, other_scaler, fz_scaler, enhanced_input_cols = enhanced_data_preprocessing(
            csv_path, input_cols, output_cols
        )

        print(f"预处理完成 | 输入维度: {X_scaled.shape[1]} | 输出维度: {y_scaled.shape[1]}")
        print(f"新增特征: {enhanced_input_cols[-4:]}")

        # 保存标准化器
        joblib.dump(x_scaler, 'enhanced_x_scaler.pkl')
        joblib.dump(other_scaler, 'enhanced_other_scaler.pkl')
        joblib.dump(fz_scaler, 'enhanced_fz_scaler.pkl')
        print("标准化器已保存")

    except Exception as e:
        print(f"数据处理错误: {str(e)}")
        return

    # 2. 创建PyTorch数据集
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    # 数据集划分 (70%-15%-15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"\n数据集划分: 训练集({len(train_dataset)}) | 验证集({len(val_dataset)}) | 测试集({test_size})")

    # 3. 创建数据加载器
    batch_size = 128  # 更大的批大小适应混合精度训练
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    # 4. 初始化模型
    model = EnhancedFzModel(
        input_size=len(enhanced_input_cols),
        output_size=len(output_cols)
    ).to(device)

    # 输出模型概况
    print(f"\n模型初始化完成，结构摘要:")
    print(model)
    print(f"参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"训练设备: {device}")

    # 5. 初始化损失函数（Fz权重设为5.0）
    loss_fn = CombinedLoss(fz_weight=5.0)
    print(f"\n损失函数配置: Fz权重 = 5.0 | 时间约束权重 = 0.1")

    # 6. 训练模型
    print("\n开始模型训练...")
    try:
        train_losses, val_losses, best_val_loss = train_enhanced_model(
            model, train_loader, val_loader, loss_fn,
            epochs=600, init_lr=0.0005
        )
        print("训练完成!")
    except Exception as e:
        print(f"训练出错: {str(e)}")
        return

    # 7. 评估最佳模型
    print("\n评估最佳模型...")
    model.load_state_dict(torch.load('best_enhanced_model.pth'))
    targets, predictions = evaluate_enhanced_model(
        model, test_loader, fz_scaler, other_scaler, output_cols
    )

    # 8. 可视化分析
    print("\n生成结果可视化...")
    plot_enhanced_results(
        train_losses, val_losses,
        targets, predictions,
        output_cols, best_val_loss
    )

    # 9. 保存最终模型
    torch.save(model.state_dict(), 'enhanced_fz_model_final.pth')
    print("\n最终模型已保存为 enhanced_fz_model_final.pth")


if __name__ == "__main__":
    main()