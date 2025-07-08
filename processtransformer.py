import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

import matplotlib
import torch
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import pip as sns
from scipy import stats
import math
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

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

# 转换为TensorFlow Dataset
if X_train_combined.size > 0:
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_combined, y_train_combined))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid_combined, y_valid_combined))

    # 批处理
    batch_size = 32
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    valid_dataset = valid_dataset.batch(batch_size)

    print("X_train shape:", X_train_combined.shape)
    print("y_train shape:", y_train_combined.shape)

else:
    raise ValueError("没有有效数据可用于训练")

# ===================== 使用文档中的Transformer模型 =====================
# 自动获取参数
num_features = X_train_combined.shape[-1]  # 输入特征维度
num_classes = len(label_encoder.classes_)  # 类别数量
max_case_length = time_steps  # 序列长度

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_a = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_b = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_a = layers.Dropout(rate)
        self.dropout_b = layers.Dropout(rate)

    def call(self, inputs, training=None):  # 修改这里，添加默认参数
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output, training=training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training=training)
        return self.layernorm_b(out_a + ffn_output)


# 修改 TokenAndPositionEmbedding 类
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):  # 只接受 maxlen 和 embed_dim
        super(TokenAndPositionEmbedding, self).__init__()
        self.dense = layers.Dense(embed_dim)  # 将连续值投影到embed_dim维度
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.dense(x)  # 处理连续值输入
        return x + positions


# 修改 get_next_activity_model 函数
def get_next_activity_model(max_case_length, num_features, output_dim,
                            embed_dim=36, num_heads=4, ff_dim=64):
    inputs = layers.Input(shape=(max_case_length, num_features))  # 3D输入

    # 修改为只传递 max_case_length 和 embed_dim
    x = TokenAndPositionEmbedding(max_case_length, embed_dim)(inputs)

    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# 创建模型时
transformer_model = get_next_activity_model(
    max_case_length=time_steps,
    num_features=num_features,  # 输入特征维度
    output_dim=num_classes,  # 输出类别数
    embed_dim=36,
    num_heads=4,
    ff_dim=64
)

# 编译模型
transformer_model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 打印模型结构
print("\nTransformer模型结构:")
transformer_model.summary()

# 训练模型
epochs = 100
print("\n开始训练...")
history = transformer_model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs
)

# 评估模型
print("\n评估结果:")
results = transformer_model.evaluate(valid_dataset, verbose=0)
print(f"验证集损失: {results[0]:.4f}, 验证集准确率: {results[1]:.4f}")

# 预测与评估
y_pred = transformer_model.predict(valid_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for x, y in valid_dataset], axis=0)

# 计算指标
print("\nTransformer评估指标:")
print(f"Accuracy: {accuracy_score(y_true, y_pred_classes):.4f}")
print(f"F1 Score: {f1_score(y_true, y_pred_classes, average='weighted'):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
conf_matrix = confusion_matrix(y_true, y_pred_classes)



# 设置标题和标签
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')


# ================== 新增功能模块 ==================
def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted')
    }
    return metrics

# 计算评估指标
metrics = calculate_metrics(y_true, y_pred_classes)

# 输出指标
print("\n综合评估指标:")
for metric, value in metrics.items():
    print(f"{metric.capitalize():<15}: {value:.4f}")