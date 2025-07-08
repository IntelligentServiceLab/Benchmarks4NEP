import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import pip as sns
from scipy import stats
import math
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns  # 确保正确导入 seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchinfo import summary
from torch import Tensor  # 添加这行导入
from zeta.nn import FeedForward, MultiQueryAttention
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


data = 'request'
f = open(fr"..\data\{data}.txt", 'r', encoding='utf-8')  # 读入文本
file_path = fr'..\data\{data}.csv'
def positional_encoding(pos, d_model):
    """
    生成正余弦位置编码
    :param pos: 位置索引
    :param d_model: 位置编码的维度
    :return: 位置编码向量
    """
    pe = torch.zeros(d_model)
    for i in range(0, d_model, 2):
        pe[i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
        if i + 1 < d_model:
            pe[i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pe

word_vec = 5
step = 4
pos_encoding_dim = 3  # 可以根据需要调整
device = "cuda"

lines = []
for line in f:  # 分别对每段分词
    temp = jieba.lcut(line)  # 结巴分词 精确模式
    words = []
    for i in temp:
        # 过滤掉所有的标点符号
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)
print(lines[0:5])  # 预览前5行分词结果

glove_path = "glove.6B.100d.txt"  # 你需要下载对应的GloVe预训练文件
word2vec_output_file = "glove.6B.100d.word2vec.txt"
glove2word2vec(glove_path, word2vec_output_file)

# 加载GloVe模型
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)




df = pd.read_csv(file_path)

print('数据基本信息：')
df.info()

# 存储有效数据的容器
X_train_list = []
X_valid_list = []
y_train_list = []
y_valid_list = []

# 新增统计变量
total_cases = 0
insufficient_data_cases = 0
error_cases = 0

# 对标签进行编码
label_encoder = LabelEncoder()
df['event'] = label_encoder.fit_transform(df['event'])  # 将事件转换为类别标签

grouped = df.groupby('case')
grouped_list = list(grouped)
train_groups, test_groups = train_test_split(grouped_list, test_size=0.3, random_state=42)
# 将训练集和测试集转换为 DataFrame
train_df = pd.concat([group for _, group in train_groups])
valid_df = pd.concat([group for _, group in test_groups])
print(df.shape)
print(train_df.shape)
print(valid_df.shape)





time_steps = step  # 定义时间步长

for case_id, group in train_df.groupby('case'):
    total_cases += 1  # 统计总案例数
    try:
        # 转换为词向量，处理未登录词
        event_data = []
        count = 1
        for pos, event in enumerate(group['event']):  # 添加位置索引
            try:
                vec = model.get_vector(event)
                # 生成位置编码
                pos_enc = positional_encoding(pos, pos_encoding_dim)
                # 将词向量和位置编码拼接
                combined_vec = np.concatenate([vec, pos_enc.numpy()])
                event_data.append(combined_vec)
                count += 1
            except KeyError:
                print(f"Case {case_id} 包含未登录词 '{event}'，已跳过")
                continue

        # 检查有效数据长度
        if len(event_data) < time_steps + 1:
            print(f"Case {case_id} 数据不足 {time_steps} 个，已跳过")
            insufficient_data_cases += 1  # 统计数据不足案例
            continue

        # 转换为numpy数组
        event_data = np.array(event_data)

        # 数据标准化
        scaler = StandardScaler()
        data = scaler.fit_transform(event_data)

        # 生成时间序列样本
        X_list = []
        y_list = []
        for i in range(len(data) - time_steps):
            X_list.append(data[i:i + time_steps])
            y_list.append(group['event'].iloc[i + time_steps])  # 使用类别标签

        X = np.array(X_list)
        y = np.array(y_list)

        # 验证形状
        if X.ndim != 3:
            print(f"Case {case_id} 数据形状异常: {X.shape}，已跳过")
            error_cases += 1  # 统计数据错误案例
            continue

        # 收集数据
        X_train_list.append(X)
        y_train_list.append(y)

    except Exception as e:
        print(f"处理 case {case_id} 时发生错误: {str(e)}")
        error_cases += 1  # 统计异常案例
        continue

# 测试集处理
for case_id, group in valid_df.groupby('case'):
    try:
        # 转换为词向量，处理未登录词
        event_data = []
        for pos, event in enumerate(group['event']):  # 添加位置索引
            try:
                vec = model.get_vector(event)
                # 生成位置编码
                pos_enc = positional_encoding(pos, pos_encoding_dim)
                # 将词向量和位置编码拼接
                combined_vec = np.concatenate([vec, pos_enc.numpy()])
                event_data.append(combined_vec)
                count += 1
            except KeyError:
                print(f"Case {case_id} 包含未登录词 '{event}'，已跳过")
                continue

        # 检查有效数据长度
        if len(event_data) < time_steps + 1:
            print(f"Case {case_id} 数据不足 {time_steps} 个，已跳过")
            continue

        # 转换为numpy数组
        event_data = np.array(event_data)

        # 数据标准化
        scaler = StandardScaler()
        data = scaler.fit_transform(event_data)

        # 生成时间序列样本
        X_list = []
        y_list = []
        for i in range(len(data) - time_steps):
            X_list.append(data[i:i + time_steps])
            y_list.append(group['event'].iloc[i + time_steps])  # 使用类别标签

        X = np.array(X_list)
        y = np.array(y_list)

        # 验证形状
        if X.ndim != 3:
            print(f"Case {case_id} 数据形状异常: {X.shape}，已跳过")
            continue

        # 收集数据
        X_valid_list.append(X)
        y_valid_list.append(y)

    except Exception as e:
        print(f"处理 case {case_id} 时发生错误: {str(e)}")
        error_cases += 1  # 统计异常案例
        continue

for case_id, group in grouped:
    total_cases += 1  # 统计总案例数

# 输出数据问题统计
print("\n数据统计信息:")
print(f"总案例数: {total_cases}")
print(f"数据不足案例数: {insufficient_data_cases} ({insufficient_data_cases / total_cases * 100:.2f}%)")
print(f"处理异常案例数: {error_cases} ({error_cases / total_cases * 100:.2f}%)")
print(f"有效案例数: {total_cases - insufficient_data_cases - error_cases}")

# 合并所有case的数据
X_train_combined = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
X_valid_combined = np.concatenate(X_valid_list, axis=0) if X_valid_list else np.array([])
y_train_combined = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
y_valid_combined = np.concatenate(y_valid_list, axis=0) if y_valid_list else np.array([])

print("\n最终数据形状:")
print("X_train:", X_train_combined.shape)
print("X_valid:", X_valid_combined.shape)
print("y_train:", y_train_combined.shape)
print("y_valid:", y_valid_combined.shape)

# 转换为Tensor
if X_train_combined.size > 0:
    X_train_tensor = torch.from_numpy(X_train_combined).float().to(device)
    X_valid_tensor = torch.from_numpy(X_valid_combined).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_combined).long().to(device)  # 使用 long 类型
    y_valid_tensor = torch.from_numpy(y_valid_combined).long().to(device)
    print("X_train_tensor shape in train_loader:", X_train_tensor.shape)
    print("y_train_tensor shape in train_loader:", y_train_tensor.shape)

else:
    raise ValueError("没有有效数据可用于训练")


class DataHandler(Dataset):
    def __init__(self, X_train_tensor, y_train_tensor, X_valid_tensor, y_valid_tensor):
        self.X_train_tensor = X_train_tensor
        self.y_train_tensor = y_train_tensor
        self.X_valid_tensor = X_valid_tensor
        self.y_valid_tensor = y_valid_tensor

    def __len__(self):
        return len(self.X_train_tensor)

    def __getitem__(self, idx):
        return self.X_train_tensor[idx], self.y_train_tensor[idx]

    def train_loader(self):
        return DataLoader(TensorDataset(self.X_train_tensor, self.y_train_tensor),
                          batch_size=32, shuffle=True)

    def valid_loader(self):
        return DataLoader(TensorDataset(self.X_valid_tensor, self.y_valid_tensor),
                          batch_size=32, shuffle=False)

    def train_loader(self):
        print("X_train_tensor shape in train_loader:", self.X_train_tensor.shape)
        print("y_train_tensor shape in train_loader:", self.y_train_tensor.shape)
        return DataLoader(TensorDataset(self.X_train_tensor, self.y_train_tensor),
                          batch_size=32, shuffle=True)


data_handler = DataHandler(X_train_tensor, y_train_tensor, X_valid_tensor, y_valid_tensor)
train_loader = data_handler.train_loader()
valid_loader = data_handler.valid_loader()


class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            load = gate_scores.sum(0)  # Sum over all examples
            importance = gate_scores.sum(1)  # Sum over all experts

            # Aux loss is mean suqared difference between load and importance
            loss = ((load - importance) ** 2).mean()

            return gate_scores, loss

        return gate_scores, None


class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                FeedForward(dim, dim, mult, *args, **kwargs)
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )

        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output, loss
class SwitchTransformerBlock(nn.Module):
    """
    SwitchTransformerBlock is a module that represents a single block of the Switch Transformer model.

    Args:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mult (int, optional): The multiplier for the hidden dimension in the feed-forward network. Defaults to 4.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        depth (int, optional): The number of layers in the block. Defaults to 12.
        num_experts (int, optional): The number of experts in the SwitchMoE layer. Defaults to 6.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mult (int): The multiplier for the hidden dimension in the feed-forward network.
        dropout (float): The dropout rate.
        attn_layers (nn.ModuleList): List of MultiQueryAttention layers.
        ffn_layers (nn.ModuleList): List of SwitchMoE layers.

    Examples:
        >>> block = SwitchTransformerBlock(dim=512, heads=8, dim_head=64)
        >>> x = torch.randn(1, 10, 512)
        >>> out = block(x)
        >>> out.shape

    """

    def __init__(
            self,
            dim: int,
            heads: int,
            dim_head: int,
            mult: int = 4,
            dropout: float = 0.1,
            num_experts: int = 3,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.dropout = dropout

        self.attn = MultiQueryAttention(
            dim, heads, qk_ln=True * args, **kwargs
        )

        self.ffn = SwitchMoE(
            dim, dim * mult, dim, num_experts, *args, **kwargs
        )

        self.add_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchTransformerBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        resi = x
        x, _, _ = self.attn(x)
        x = x + resi
        x = self.add_norm(x)
        add_normed = x

        ##### MoE #####
        x, _ = self.ffn(x)
        x = x + add_normed
        x = self.add_norm(x)
        return x

class SwitchTransformer(nn.Module):
    """
    修改后的SwitchTransformer适配时序预测任务
    """

    def __init__(
            self,
            input_dim: int,
            dim: int,
            heads: int,
            num_classes: int,
            dim_head: int = 64,
            mult: int = 4,
            dropout: float = 0.1,
            num_experts: int = 3,
            depth: int = 4,
            *args,
            ** kwargs,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, dim)

        # 创建Transformer堆栈
        self.layers = nn.ModuleList([
            SwitchTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mult=mult,
                dropout=dropout,
                num_experts=num_experts,
                *args,
        ** kwargs
        ) for _ in range(depth)
        ])

        # 输出层
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        # 输入形状: (batch, seq_len, features)
        x = self.projection(x)  # 投影到模型维度

        # 通过所有Transformer层
        for layer in self.layers:
            x = layer(x)

        # 聚合时序信息（取最后一个时间步）
        x = x[:, -1, :]  # 改为平均池化可以使用x.mean(dim=1)

        return self.to_out(x)


# 自动获取特征维度
num_features = X_train_combined.shape[-1]  # 原始特征维度
num_classes = len(label_encoder.classes_)


# 自动获取特征维度
num_features = X_train_combined.shape[-1]
num_classes = len(label_encoder.classes_)

# 替换原始模型定义部分
model = SwitchTransformer(
    input_dim=num_features,  # 输入特征维度
    dim=64,                 # 模型隐藏维度
    heads=4,                # 注意力头数
    num_classes=num_classes, # 输出类别数
    dim_head=32,            # 每个头的维度
    mult=4,                 # FFN扩展倍数
    dropout=0.1,            # 丢弃率
    num_experts=2,          # 每个MoE层的专家数
    depth=3                 # Transformer层数
).to(device)

# 修改优化器配置（Transformer通常需要更稳定的训练）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("\n模型结构:")
summary(model, input_size=(32, time_steps, num_features))


def train(model, loader, optimizer):
    model.train()
    total_loss, total_acc = 0, 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (outputs.argmax(dim=1) == y_batch).float().mean().item()
    return total_loss / len(loader), total_acc / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            total_acc += (outputs.argmax(dim=1) == y_batch).float().mean().item()
    return total_loss / len(loader), total_acc / len(loader)


epochs = 100
print("\n开始训练...")
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer)
    valid_loss, valid_acc = evaluate(model, valid_loader)

    print(f"Epoch {epoch + 1:03}/{epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}")

# 预测与评估
targets, predictions = [], []
with torch.no_grad():
    for X_batch, y_batch in valid_loader:
        outputs = model(X_batch)
        targets.extend(y_batch.cpu().numpy())
        predictions.extend(outputs.argmax(dim=1).cpu().numpy())

# 计算指标
print("\nswitch-transformer评估指标:")
print(f"Accuracy: {accuracy_score(targets, predictions):.4f}")
print(f"F1 Score: {f1_score(targets, predictions, average='weighted'):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(targets, predictions))
conf_matrix = confusion_matrix(targets, predictions)

# 创建热力图
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)

# 设置标题和标签
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
# ================== 新增功能模块 ==================
import os
from sklearn.metrics import precision_score, recall_score


def get_model_name(model):
    """自动从模型类名获取模型名称"""
    return model.__class__.__name__


def get_dataset_name(file_path):
    """从文件路径提取数据集名称"""
    return os.path.splitext(os.path.basename(file_path))[0]


def save_confusion_matrix(conf_matrix, model_name, dataset_name,
                          class_names, save_path):
    """保存带标题的混淆矩阵"""
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names)

    # 设置标题
    main_title = f"{model_name} + {dataset_name}"
    plt.suptitle(main_title, y=1.02, fontsize=14)
    plt.title("Confusion Matrix", fontsize=12)

    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    filename = f"{main_title.replace('+', '_')}.png"
    save_file = os.path.join(save_path, filename)

    plt.savefig(save_file, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至: {save_file}")


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted')
    }
    return metrics


# ================== 应用功能模块 ==================
# 自动获取名称
model_name = get_model_name(model)
dataset_name = get_dataset_name(file_path)

# 计算评估指标
metrics = calculate_metrics(targets, predictions)

# 输出指标
print("\n综合评估指标:")
for metric, value in metrics.items():
    print(f"{metric.capitalize():<15}: {value:.4f}")

