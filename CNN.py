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
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

data = 'BPIC_2012_W'
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


# 修改训练集处理逻辑
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



class CNN_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN_Model, self).__init__()
        # 输入形状：(batch_size, time_steps, input_dim)
        # 调整为CNN需要的形状：(batch_size, input_dim, time_steps)

        # 第一层卷积
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,  # 输入特征维度作为通道数
            out_channels=64,  # 输出通道数
            kernel_size=3,  # 卷积核大小
            padding=1  # 保持时间维度长度
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 池化后时间维度减半

        # 第二层卷积
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 全连接层
        self.fc = nn.Linear(128 * (time_steps // 4), output_dim)  # 根据池化次数调整

    def forward(self, x):
        # 调整输入维度 [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)

        # 第一层卷积
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 展平特征
        x = x.view(x.size(0), -1)

        # 全连接层
        return self.fc(x)


# 自动获取特征维度
num_features = X_train_combined.shape[-1]
num_classes = len(label_encoder.classes_)

# 替换原始模型定义部分
model = CNN_Model(
    input_dim=num_features,
    output_dim=num_classes
).to(device)

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
print("\nBiLSTM + Transformer评估指标:")
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

