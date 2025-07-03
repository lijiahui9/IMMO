#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as tfkl
import sys
import scipy as sp
import scipy.sparse
import h5py
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import History
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt


# In[2]:


met_df = pd.read_csv(r"C:\Users\LJH\Desktop\论文\1\metabolomic_abundance.maxmin.flt.csv", index_col=0)
tra_df = pd.read_csv(r"C:\Users\LJH\Desktop\论文\1\metatranscriptome_abundance.maxmin.flt.csv", index_col=0)
gen_df = pd.read_csv(r"C:\Users\LJH\Desktop\论文\1\microbiome_abundance.maxmin.flt.csv", index_col=0)
file_path = r"C:\Users\LJH\Desktop\论文\1\ID_processed.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
cell_types = df['diagnosis'].astype(str)
cell_ids=df.index
print(cell_types)
print(cell_ids)

x=gen_df
y=tra_df
z=met_df
x = x.drop_duplicates(keep='first')
y = y.drop_duplicates(keep='first')
z = z.drop_duplicates(keep='first')
x.index = x.index.astype(str)
x.index = x.index.str.lower()
x.index = x.index.str.strip()
y.index = y.index.astype(str)
y.index = y.index.str.lower()
y.index = y.index.str.strip()
z.index = z.index.astype(str)
z.index = z.index.str.lower()
z.index = z.index.str.strip()

all_rows = set(x.index).union(set(y.index)).union(set(z.index))

x_re0 = x.reindex(all_rows, fill_value=np.nan)
y_re0 = y.reindex(all_rows, fill_value=np.nan)
z_re0 = z.reindex(all_rows, fill_value=np.nan)

with open('sampe_name_output.txt', 'w') as file:
    for item in all_rows:
        file.write(item + '\n')

result = pd.concat([x_re0, y_re0, z_re0], ignore_index=True)

x_m = x_re0.notna().astype(int)
y_m = y_re0.notna().astype(int)
z_m = z_re0.notna().astype(int)

x_re0.fillna(0, inplace=True)
y_re0.fillna(0, inplace=True)
z_re0.fillna(0, inplace=True)

x_re1=x_re0.values
y_re1=y_re0.values
z_re1=z_re0.values

x_m=x_m.values
y_m=y_m.values
z_m=z_m.values


# # Static Mask

# In[11]:


def MaskGenerator(inputs, p):
    mask = np.random.choice(2, size=inputs.shape, p=[1 - p, p]).astype(tf.keras.backend.floatx())
    return mask
class MultiModalAutoencoder(tf.keras.Model):
    def __init__(self, input_dim_x, input_dim_y, input_dim_z, latent_dim=32, dropout_rate=0.1):
        super(MultiModalAutoencoder, self).__init__()

        self.encoder_x = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim_x,)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.encoder_y = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim_y,)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.encoder_z = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim_z,)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate)
        ])

        self.fusion_layer = tf.keras.layers.Dense(latent_dim, activation='relu')
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(input_dim_x + input_dim_y + input_dim_z, activation='linear')
        ])

    def get_fused_encoding(self, x, y, z):
        """直接使用原始数据获取融合编码，不应用任何掩码"""
        encoded_x = self.encoder_x(x)
        encoded_y = self.encoder_y(y)
        encoded_z = self.encoder_z(z)
        concatenated = tf.keras.layers.Concatenate()([encoded_x, encoded_y, encoded_z])
        return self.fusion_layer(concatenated)

    def call(self, inputs):
        """前向传播时接收已经掩码处理的数据"""
        x, y, z = inputs
        encoded_x = self.encoder_x(x)
        encoded_y = self.encoder_y(y)
        encoded_z = self.encoder_z(z)
        concatenated = tf.keras.layers.Concatenate()([encoded_x, encoded_y, encoded_z])
        fused = self.fusion_layer(concatenated)
        return self.decoder(fused)
    
def masked_mse_loss(y_true, y_pred, masks, weights=(0.3, 0.3, 0.4)):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    masks = tf.cast(masks, tf.float32)

    y_true_masked = y_true * masks
    y_pred_masked = y_pred * masks

    y_true_x, y_true_y, y_true_z = tf.split(y_true, [input_dim_x, input_dim_y, input_dim_z], axis=1)
    y_pred_x, y_pred_y, y_pred_z = tf.split(y_pred, [input_dim_x, input_dim_y, input_dim_z], axis=1)
    mask_x, mask_y, mask_z = tf.split(masks, [input_dim_x, input_dim_y, input_dim_z], axis=1)

    mse_x = tf.reduce_mean(tf.square((y_true_x - y_pred_x) * mask_x))
    mse_y = tf.reduce_mean(tf.square((y_true_y - y_pred_y) * mask_y))
    mse_z = tf.reduce_mean(tf.square((y_true_z - y_pred_z) * mask_z))

    total_loss = weights[0]*mse_x + weights[1]*mse_y + weights[2]*mse_z
    return total_loss

def train_step(x_batch, y_batch, z_batch, x_m_batch, y_m_batch, z_m_batch, optimizer):
    with tf.GradientTape() as tape:
        mask_x = MaskGenerator(x_batch, p=0.75)
        mask_y = MaskGenerator(y_batch, p=0.75)
        mask_z = MaskGenerator(z_batch, p=0.75)
    
        x_masked = x_batch * mask_x
        y_masked = y_batch * mask_y
        z_masked = z_batch * mask_z
        predictions = multi_modal_autoencoder((x_masked, y_masked, z_masked))
        y_true = tf.concat([x_batch, y_batch, z_batch], axis=1)
        masks = tf.concat([x_m_batch, y_m_batch, z_m_batch], axis=1)
    
        loss = masked_mse_loss(y_true, predictions, masks)
    
    gradients = tape.gradient(loss, multi_modal_autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, multi_modal_autoencoder.trainable_variables))
    return loss
# 添加早停策略
# 早停策略
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, warmup=5):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def on_epoch_end(self, epoch, current_loss):
        if epoch < self.warmup:
            return
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

# 初始化早停策略
early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

# 初始化模型和优化器
input_dim_x = x_re0.shape[1]
input_dim_y = y_re0.shape[1]
input_dim_z = z_re0.shape[1]
latent_dim = 110
multi_modal_autoencoder = MultiModalAutoencoder(input_dim_x, input_dim_y, input_dim_z, latent_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

x_train = x_re1
x_m_train = x_m
y_train = y_re1
y_m_train = y_m
z_train = z_re1
z_m_train = z_m
# 直接使用全部数据进行训练
num_epochs = 100
batch_size = 64
train_loss_history = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, len(x_train), batch_size):
        # 获取当前batch数据
        end_idx = min(i + batch_size, len(x_train))
        x_batch = x_train[i:end_idx]
        y_batch = y_train[i:end_idx]
        z_batch = z_train[i:end_idx]
        x_m_batch = x_m_train[i:end_idx]
        y_m_batch = y_m_train[i:end_idx]
        z_m_batch = z_m_train[i:end_idx]

        # 执行训练步骤
        batch_loss = train_step(x_batch, y_batch, z_batch, x_m_batch, y_m_batch, z_m_batch, optimizer)
        epoch_loss += batch_loss.numpy()
        
        # 打印批次信息
        print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | Batch {i//batch_size:04d} | Loss: {batch_loss:.4f}", end='\r')
    
    # 计算并存储平均损失
    avg_loss = epoch_loss / (len(x_train) // batch_size + 1)
    train_loss_history.append(avg_loss)
    
    # 打印epoch信息
    print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | Average Loss: {avg_loss:.4f}")
    
    # 检查早停条件
    early_stopping.on_epoch_end(epoch, avg_loss)
    if early_stopping.stop_training:
        print(f"\nEarly stopping triggered at epoch {epoch+1}!")
        break

# 绘制损失函数图，更加规范美观
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='Training Loss', color='blue', linewidth=2)
plt.title('Training Loss History', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('training_loss.png')  # 保存图像
plt.show()

# 使用完整数据生成潜在表示（不应用任何掩码）
latent_representations = multi_modal_autoencoder.get_fused_encoding(x_re1, y_re1, z_re1).numpy()
latent_df = pd.DataFrame(latent_representations, index=x_re0.index[-len(latent_representations):])
latent_df.to_csv('latent_representation_110_static.csv', index=True)


# # Dynamic Mask

# In[13]:


# 动态掩码生成器
class DynamicMaskGenerator:
    def __init__(self, initial_p=0.75, growth_rate=0.95, max_p=0.9):
        self.initial_p = initial_p
        self.growth_rate = growth_rate  
        self.max_p = max_p

    def generate_mask(self, inputs, epoch):
        """生成动态掩码，p值随epoch指数增长"""
        # 计算当前epoch的保留概率
        p = self.initial_p + (self.max_p - self.initial_p) * (1 - self.growth_rate ** epoch)
        p = min(p, self.max_p)  # 确保不超过最大阈值
        # 生成二值掩码
        mask = np.random.choice(2, size=inputs.shape, p=[1 - p, p]).astype(tf.keras.backend.floatx())
        return mask

# 多模态自编码器
class MultiModalAutoencoder(models.Model):
    def __init__(self, input_dim_x, input_dim_y, input_dim_z, latent_dim=110, dropout_rate=0.473):
        super(MultiModalAutoencoder, self).__init__()
        
        def build_encoder(input_dim):
            return models.Sequential([
                layers.Dense(256, activation='swish', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                layers.Dense(128, activation='swish'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                layers.Dense(64, activation='swish'),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate)
            ])
        
        self.encoder_x = build_encoder(input_dim_x)
        self.encoder_y = build_encoder(input_dim_y)
        self.encoder_z = build_encoder(input_dim_z)
        
        self.fusion_layer = models.Sequential([
            layers.Dense(128, activation='swish'),
            layers.Dropout(dropout_rate),
            layers.Dense(latent_dim, activation='relu')
        ])
        
        self.decoder = models.Sequential([
            layers.Dense(128, activation='swish', input_shape=(latent_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='swish'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(input_dim_x + input_dim_y + input_dim_z, activation='linear')
        ])

    def get_fused_encoding(self, x, y, z):
        encoded_x = self.encoder_x(x)
        encoded_y = self.encoder_y(y)
        encoded_z = self.encoder_z(z)
        concatenated = layers.Concatenate()([encoded_x, encoded_y, encoded_z])
        return self.fusion_layer(concatenated)

    def call(self, inputs):
        x, y, z = inputs
        return self.decoder(self.get_fused_encoding(x, y, z))

# 损失函数
def masked_loss(y_true, y_pred, masks, weights=(0.3, 0.3, 0.4)):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    masks = tf.cast(masks, tf.float32)

    # 分割各模态数据
    y_true_x, y_true_y, y_true_z = tf.split(y_true, [input_dim_x, input_dim_y, input_dim_z], axis=1)
    y_pred_x, y_pred_y, y_pred_z = tf.split(y_pred, [input_dim_x, input_dim_y, input_dim_z], axis=1)
    mask_x, mask_y, mask_z = tf.split(masks, [input_dim_x, input_dim_y, input_dim_z], axis=1)

    # 应用掩码并计算各模态损失
    mse_x = tf.reduce_mean(tf.square((y_true_x * mask_x - y_pred_x * mask_x)))
    mse_y = tf.reduce_mean(tf.square((y_true_y * mask_y - y_pred_y * mask_y)))
    mse_z = tf.reduce_mean(tf.square((y_true_z * mask_z - y_pred_z * mask_z)))

    # 加权组合总损失
    total_loss = weights[0]*mse_x + weights[1]*mse_y + weights[2]*mse_z
    return total_loss

# 训练步骤
def train_step(x_batch, y_batch, z_batch, x_m_batch, y_m_batch, z_m_batch,
              mask_generator, model, optimizer, epoch):
    with tf.GradientTape() as tape:
        mask_x = mask_generator.generate_mask(x_batch, epoch)
        mask_y = mask_generator.generate_mask(y_batch, epoch)
        mask_z = mask_generator.generate_mask(z_batch, epoch)
        
        x_masked = x_batch * mask_x
        y_masked = y_batch * mask_y
        z_masked = z_batch * mask_z
        
        predictions = model((x_masked, y_masked, z_masked))
        y_true = tf.concat([x_batch, y_batch, z_batch], axis=1)
        masks = tf.concat([x_m_batch, y_m_batch, z_m_batch], axis=1)
        loss = masked_loss(y_true, predictions, masks)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 早停策略
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, warmup=5):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def on_epoch_end(self, epoch, current_loss):
        if epoch < self.warmup:
            return
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

# 训练函数
def train_model(x_train, y_train, z_train, x_m_train, y_m_train, z_m_train,
               input_dims, latent_dim=110, epochs=100, batch_size=64):
    
    # 初始化动态掩码生成器
    mask_generator = DynamicMaskGenerator(
        initial_p=0.75,       # 初始保留概率
        growth_rate=0.95,     # 增长率
        max_p=0.9             # 最大保留概率
    )
    
    model = MultiModalAutoencoder(*input_dims, latent_dim=latent_dim, dropout_rate=0.473)
    
    # 学习率调度
    lr_scheduler = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0018379230255141633,
        decay_steps=200,
        decay_rate=0.96,
        staircase=True)
    optimizer = optimizers.Adam(lr_scheduler)
    
    early_stopping = EarlyStopping()
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train, z_train, x_m_train, y_m_train, z_m_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_loss_history = []
    best_weights = None
    
    for epoch in range(epochs):
        epoch_losses = []
        dataset = dataset.shuffle(buffer_size=1024)
        
        for batch in dataset:
            x_b, y_b, z_b, xm_b, ym_b, zm_b = batch
            loss = train_step(x_b, y_b, z_b, xm_b, ym_b, zm_b,
                             mask_generator, model, optimizer, epoch)
            epoch_losses.append(loss.numpy())
        
        avg_loss = np.mean(epoch_losses)
        train_loss_history.append(avg_loss)
        
        early_stopping.on_epoch_end(epoch, avg_loss)
        if early_stopping.stop_training:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
        
        if avg_loss == min(train_loss_history):
            best_weights = model.get_weights()
        
        current_lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.5f}")
    
    if best_weights:
        model.set_weights(best_weights)
    
    # 绘制损失函数图，更加规范美观
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='Training Loss', color='blue', linewidth=2)
    plt.title('Training Loss History', fontsize=18)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.minorticks_on()
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300)  
    plt.show()
    
    return model

if __name__ == "__main__":
    os.chdir(r"C:\Users\LJH\Desktop\论文\1")
    
    input_dim_x = x_re0.shape[1]
    input_dim_y = y_re0.shape[1]
    input_dim_z = z_re0.shape[1]
    input_dims = (input_dim_x, input_dim_y, input_dim_z)
    
    trained_model = train_model(
        x_re1, y_re1, z_re1,
        x_m, y_m, z_m,
        input_dims=input_dims,
        latent_dim=110,
        epochs=100,
        batch_size=64
    )
    
    latent_rep = trained_model.get_fused_encoding(x_re1, y_re1, z_re1).numpy()
    latent_df = pd.DataFrame(latent_rep, index=x_re0.index[-len(latent_rep):])
    latent_df.to_csv('latent_representation_110_dynamic.csv', index=True)


# In[14]:


import matplotlib.pyplot as plt

# Dynamic Mask 损失数据
dynamic_mask_loss = [
    0.0040, 0.0033, 0.0032, 0.0029, 0.0027, 0.0026, 0.0025, 0.0025, 0.0024, 0.0024,
    0.0023, 0.0022, 0.0022, 0.0021, 0.0020, 0.0020, 0.0020, 0.0020, 0.0019, 0.0019,
    0.0018, 0.0018, 0.0017, 0.0017, 0.0017, 0.0017, 0.0016, 0.0016, 0.0016, 0.0016,
    0.0016, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0014, 0.0014, 0.0014, 0.0014,
    0.0014, 0.0014, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013,
    0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0011,
    0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
    0.0010, 0.0010, 0.0010
]

# Static Mask 损失数据
static_mask_loss = [
    0.0035, 0.0028, 0.0025, 0.0025, 0.0024, 0.0024, 0.0025, 0.0023, 0.0023, 0.0023,
    0.0022, 0.0022, 0.0022, 0.0021, 0.0022, 0.0021, 0.0020, 0.0020, 0.0019, 0.0020,
    0.0020, 0.0019, 0.0021, 0.0020, 0.0020
]

# 绘制图像
epochs_dynamic = range(1, len(dynamic_mask_loss) + 1)
epochs_static = range(1, len(static_mask_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs_dynamic, dynamic_mask_loss, label='Dynamic Mask Loss', color='blue')
plt.plot(epochs_static, static_mask_loss, label='Static Mask Loss', color='red')

# 标注Early stopping点
plt.axvline(x=73, color='blue', linestyle='--', label='Dynamic Mask Early Stopping')
plt.axvline(x=24, color='red', linestyle='--', label='Static Mask Early Stopping')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison Between Dynamic and Static Masks')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




