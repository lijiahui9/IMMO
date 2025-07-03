#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.models import Model 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

file_paths = {
    'RNA': r"C:\Users\LJH\Desktop\论文\Prediabetes\RNAseq_filtered_reduced1.xlsx",
    'proteome': r'C:\Users\LJH\Desktop\论文\Prediabetes数据集\proteome_abundance.csv',
    'metabolome': r"C:\Users\LJH\Desktop\论文\Prediabetes数据集\metabolome_abundance.csv",
    'gut_16s': r"C:\Users\LJH\Desktop\论文\Prediabetes数据集\gut_16s_abundance.csv"
}

dataframes = {}
for name, path in file_paths.items():
    if path.endswith('.xlsx'):
        df = pd.read_excel(path, index_col=0)  
    else: 
        df = pd.read_csv(path, index_col=0)  
    print(f"Shape of {name}: {df.shape}")
    dataframes[name] = df

def preprocess_and_log_normalize(df):
    df = df[~df.index.duplicated(keep='first')] 
    df.index = df.index.astype(str).str.lower().str.strip()
    epsilon = 1e-100  
    df[df.select_dtypes(include=[np.number]).columns] = \
        df.select_dtypes(include=[np.number]).add(epsilon).apply(np.log1p)
    return df

processed_dfs = {name: preprocess_and_log_normalize(df) for name, df in dataframes.items()}

all_rows = sorted(set.union(*[set(df.index) for df in processed_dfs.values()]))
reindexed_dfs = {}
masks = {}


for name, df in processed_dfs.items():
    reindexed_df = df.reindex(all_rows, fill_value=np.nan)
    mask = reindexed_df.notna().astype(int)
    reindexed_df_filled = reindexed_df.fillna(0)
    reindexed_dfs[name] = reindexed_df_filled
    masks[name] = mask

rna_re1 = reindexed_dfs['RNA'].values
proteome_re1 = reindexed_dfs['proteome'].values
metabolome_re1 = reindexed_dfs['metabolome'].values
gut_16s_re1 = reindexed_dfs['gut_16s'].values

rna_m = masks['RNA'].values
proteome_m = masks['proteome'].values
metabolome_m = masks['metabolome'].values
gut_16s_m = masks['gut_16s'].values

scaler = StandardScaler()
rna_re1 = scaler.fit_transform(rna_re1)
proteome_re1 = scaler.fit_transform(proteome_re1)
metabolome_re1 = scaler.fit_transform(metabolome_re1)
gut_16s_re1 = scaler.fit_transform(gut_16s_re1)


# In[12]:


class DynamicMaskGenerator:
    def __init__(self, initial_p=0.75, growth_rate=0.95, max_p=0.9):
        self.initial_p = initial_p
        self.growth_rate = growth_rate  
        self.max_p = max_p

    def generate_mask(self, inputs, epoch):
        """生成动态掩码，p值随epoch指数增长"""
        p = self.initial_p + (self.max_p - self.initial_p) * (1 - self.growth_rate ** epoch)
        p = min(p, self.max_p)  
        mask = np.random.choice(2, size=inputs.shape, p=[1 - p, p]).astype(tf.keras.backend.floatx())
        return mask

class MultiModalAutoencoder(models.Model):
    def __init__(self, input_dim_rna, input_dim_proteome, input_dim_metabolome, input_dim_gut_16s, latent_dim=110, dropout_rate=0.473):
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
        
        self.encoder_rna = build_encoder(input_dim_rna)
        self.encoder_proteome = build_encoder(input_dim_proteome)
        self.encoder_metabolome = build_encoder(input_dim_metabolome)
        self.encoder_gut_16s = build_encoder(input_dim_gut_16s)
        
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
            layers.Dense(input_dim_rna + input_dim_proteome + input_dim_metabolome + input_dim_gut_16s, activation='linear')
        ])

    def get_fused_encoding(self, rna, proteome, metabolome, gut_16s):
        encoded_rna = self.encoder_rna(rna)
        encoded_proteome = self.encoder_proteome(proteome)
        encoded_metabolome = self.encoder_metabolome(metabolome)
        encoded_gut_16s = self.encoder_gut_16s(gut_16s)
        concatenated = layers.Concatenate()([encoded_rna, encoded_proteome, encoded_metabolome, encoded_gut_16s])
        return self.fusion_layer(concatenated)

    def call(self, inputs):
        rna, proteome, metabolome, gut_16s = inputs
        return self.decoder(self.get_fused_encoding(rna, proteome, metabolome, gut_16s))

# 损失函数
def masked_loss(y_true, y_pred, masks, weights=(0.2, 0.2, 0.25, 0.35)):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    masks = tf.cast(masks, tf.float32)

    y_true_rna, y_true_proteome, y_true_metabolome, y_true_gut_16s = tf.split(
        y_true, [input_dim_rna, input_dim_proteome, input_dim_metabolome, input_dim_gut_16s], axis=1)
    y_pred_rna, y_pred_proteome, y_pred_metabolome, y_pred_gut_16s = tf.split(
        y_pred, [input_dim_rna, input_dim_proteome, input_dim_metabolome, input_dim_gut_16s], axis=1)
    mask_rna, mask_proteome, mask_metabolome, mask_gut_16s = tf.split(
        masks, [input_dim_rna, input_dim_proteome, input_dim_metabolome, input_dim_gut_16s], axis=1)

    mse_rna = tf.reduce_mean(tf.square((y_true_rna * mask_rna - y_pred_rna * mask_rna)))
    mse_proteome = tf.reduce_mean(tf.square((y_true_proteome * mask_proteome - y_pred_proteome * mask_proteome)))
    mse_metabolome = tf.reduce_mean(tf.square((y_true_metabolome * mask_metabolome - y_pred_metabolome * mask_metabolome)))
    mse_gut_16s = tf.reduce_mean(tf.square((y_true_gut_16s * mask_gut_16s - y_pred_gut_16s * mask_gut_16s)))

    total_loss = (
        weights[0] * mse_rna +
        weights[1] * mse_proteome +
        weights[2] * mse_metabolome +
        weights[3] * mse_gut_16s
    )
    return total_loss

def train_step(rna_batch, proteome_batch, metabolome_batch, gut_16s_batch,
               rna_m_batch, proteome_m_batch, metabolome_m_batch, gut_16s_m_batch,
               mask_generator, model, optimizer, epoch):
    with tf.GradientTape() as tape:
        mask_rna = mask_generator.generate_mask(rna_batch, epoch)
        mask_proteome = mask_generator.generate_mask(proteome_batch, epoch)
        mask_metabolome = mask_generator.generate_mask(metabolome_batch, epoch)
        mask_gut_16s = mask_generator.generate_mask(gut_16s_batch, epoch)
        
        rna_masked = rna_batch * mask_rna
        proteome_masked = proteome_batch * mask_proteome
        metabolome_masked = metabolome_batch * mask_metabolome
        gut_16s_masked = gut_16s_batch * mask_gut_16s
        
        predictions = model((rna_masked, proteome_masked, metabolome_masked, gut_16s_masked))
        y_true = tf.concat([rna_batch, proteome_batch, metabolome_batch, gut_16s_batch], axis=1)
        masks = tf.concat([rna_m_batch, proteome_m_batch, metabolome_m_batch, gut_16s_m_batch], axis=1)
        loss = masked_loss(y_true, predictions, masks)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

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

def train_model(rna_train, proteome_train, metabolome_train, gut_16s_train,
                rna_m_train, proteome_m_train, metabolome_m_train, gut_16s_m_train,
                input_dims, latent_dim=110, epochs=100, batch_size=64):
    
    mask_generator = DynamicMaskGenerator(
        initial_p=0.75,       
        growth_rate=0.95,     
        max_p=0.9             
    )
    
    model = MultiModalAutoencoder(*input_dims, latent_dim=latent_dim, dropout_rate=0.473)
    lr_scheduler = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0018379230255141633,
        decay_steps=200,
        decay_rate=0.96,
        staircase=True)
    optimizer = optimizers.Adam(lr_scheduler)
    
    early_stopping = EarlyStopping()
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (rna_train, proteome_train, metabolome_train, gut_16s_train,
         rna_m_train, proteome_m_train, metabolome_m_train, gut_16s_m_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_loss_history = []
    best_weights = None
    
    for epoch in range(epochs):
        epoch_losses = []
        dataset = dataset.shuffle(buffer_size=1024)
        
        for batch in dataset:
            rna_b, proteome_b, metabolome_b, gut_16s_b, rna_m_b, proteome_m_b, metabolome_m_b, gut_16s_m_b = batch
            loss = train_step(rna_b, proteome_b, metabolome_b, gut_16s_b,
                             rna_m_b, proteome_m_b, metabolome_m_b, gut_16s_m_b,
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
    
    data = {
        'rna': rna_re1,
        'proteome': proteome_re1,
        'metabolome': metabolome_re1,
        'gut_16s': gut_16s_re1,
    }
    
    input_dim_rna = data['rna'].shape[1]
    input_dim_proteome = data['proteome'].shape[1]
    input_dim_metabolome = data['metabolome'].shape[1]
    input_dim_gut_16s = data['gut_16s'].shape[1]
    input_dims = (input_dim_rna, input_dim_proteome, input_dim_metabolome, input_dim_gut_16s)
    
    trained_model = train_model(
        data['rna'], data['proteome'], data['metabolome'], data['gut_16s'],
        rna_m, proteome_m, metabolome_m, gut_16s_m,
        input_dims=input_dims,
        latent_dim=110,
        epochs=100,
        batch_size=64
    )


# In[13]:


latent_rep = trained_model.get_fused_encoding(data['rna'], data['proteome'], data['metabolome'], data['gut_16s']).numpy()


# In[15]:


if len(latent_rep) != len(all_rows):
    raise ValueError(f"latent_rep 的行数 ({len(latent_rep)}) 与 all_rows 的长度 ({len(all_rows)}) 不匹配！")
latent_df = pd.DataFrame(latent_rep, index=all_rows)
latent_df.to_csv('latent_110_diabetes.csv', index=True)
print("All rows (union of indices):", len(all_rows))
for name, df in reindexed_dfs.items():
    print(f"Reindexed DataFrame '{name}' shape: {df.shape}")
    print(f"Mask for '{name}' shape: {masks[name].shape}")
print("Latent DataFrame shape:", latent_df.shape)


# In[2]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
import random
from itertools import combinations

dataframes = {}
for name, path in file_paths.items():
    df = pd.read_csv(path, index_col=0) if name != 'RNA' else pd.read_excel(path, index_col=0)
    df.index = df.index.astype(str).str.lower().str.strip()  
    df = df[~df.index.duplicated(keep='first')] 
    dataframes[name] = df

X = pd.read_csv(r"C:\Users\LJH\Desktop\论文\1\latent_110_diabetes.csv", index_col=1)
X = pd.DataFrame(X)
X.set_index('samples', inplace=True)


# In[3]:


def extract_indices(*dfs):
    return set.intersection(*(set(df.index) for df in dfs))

def build_combination_name(keys, comb):
    df_to_key = {id(df): key for key, df in zip(keys, dataframes.values())}
    return '_'.join([df_to_key[id(df)] for df in comb])

datasets = {}
common_indices_set = extract_indices(*dataframes.values())
common_indices_list = list(common_indices_set)
datasets['common'] = X.loc[X.index.intersection(pd.Index(common_indices_list))]

for comb in combinations(dataframes.values(), 2):
    common = extract_indices(*comb)
    common_list = list(common)
    datasets[f"{build_combination_name(list(dataframes.keys()), comb)}_common"] = X.loc[X.index.intersection(pd.Index(common_list))]

for comb in combinations(dataframes.values(), 3):
    common = extract_indices(*comb)
    common_list = list(common)
    datasets[f"{build_combination_name(list(dataframes.keys()), comb)}_common"] = X.loc[X.index.intersection(pd.Index(common_list))]

datasets['all_common'] = X.loc[X.index.intersection(pd.Index(common_indices_list))]

for comb in combinations(dataframes.values(), 2):
    union = set.union(*(set(df.index) for df in comb))
    union_list = list(union)
    datasets[f"{build_combination_name(list(dataframes.keys()), comb)}_all"] = X.loc[X.index.intersection(pd.Index(union_list))]
    
for comb in combinations(dataframes.values(), 3):
    union = set.union(*(set(df.index) for df in comb))
    union_list = list(union)
    datasets[f"{build_combination_name(list(dataframes.keys()), comb)}_all"] = X.loc[X.index.intersection(pd.Index(union_list))]
union_all_set = set.union(*(set(df.index) for df in dataframes.values()))
union_all_list = list(union_all_set)
datasets['all_all'] = X.loc[X.index.intersection(pd.Index(union_all_list))]


for key, dataset in datasets.items():
    print(f"Dataset {key} has shape: {dataset.shape}")


# In[4]:


file_path = r"C:\Users\LJH\Desktop\论文\Prediabetes数据集\41586_2019_1236_MOESM3_ESM.xlsx"
label_df = pd.read_excel(file_path, sheet_name='S1_Subjects')
label_df.set_index('SubjectID', inplace=True)
label_df.index = label_df.index.astype(str).str.strip().str.lower()
print(label_df.head())


# In[5]:


def get_labels(X, label_df):
    # 确保 label_df 的索引是字符串类型，且无前后空格，全部小写
    if not isinstance(label_df.index, pd.core.indexes.base.Index):
        label_df = label_df.copy()
        label_df.index = label_df.index.astype(str).str.strip().str.lower()

    y = []
    for idx in X.index:
        # 假设 SampleID 和附加信息是由连字符分隔的
        label_prefix = idx.split('-')[0].lower().strip()  # 获取 SampleID 部分并格式化
        if label_prefix in label_df.index:
            matching_label = label_df.at[label_prefix, 'Class']
            if matching_label == "Crossover":
                label = np.nan  
            elif matching_label == "Control":
                label = 0  
            else:
                label = 1 
        else:
            label = np.nan 
        y.append(label)
    return pd.Series(y, index=X.index)

labels = get_labels(X, label_df)


# In[6]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
import matplotlib.pyplot as plt

def extract_indices(*dfs):
    return list(set.intersection(*(set(df.index.astype(str)) for df in dfs)))

def build_combination_name(keys, comb):
    df_to_key = {id(df): key for key, df in zip(keys, dataframes.values())}
    return '_'.join([df_to_key[id(df)] for df in comb])

def train_and_evaluate(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'SVM': (SVC(probability=True, random_state=42), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }),
        'GBDT': (GradientBoostingClassifier(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }),
        'Neural Network': (MLPClassifier(max_iter=500, random_state=42), {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        }),
        'LDA': (LinearDiscriminantAnalysis(), {}),
        'Ridge Regression': (RidgeClassifierCV(), {}),
        'RF': (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        })
    }
    
    model_metrics = {}
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 1))]
    
    for idx, (model_name, (model, param_grid)) in enumerate(models.items()):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        if hasattr(best_model, "predict_proba"):
            probas = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model, "decision_function"): 
            scores = best_model.decision_function(X_test)
            probas = (scores - scores.min()) / (scores.max() - scores.min())
        else: 
            probas = best_model.predict(X_test)
    
        auc = roc_auc_score(y_test, probas)
        acc = accuracy_score(y_test, best_model.predict(X_test))
        f1 = f1_score(y_test, best_model.predict(X_test))
        
        model_metrics[model_name] = {
            'AUC': round(auc, 4),
            'ACC': round(acc, 4),
            'F1': round(f1, 4)
        }

        fpr, tpr, _ = roc_curve(y_test, probas)
        plt.plot(fpr, tpr, label=f'{dataset_name} - {model_name} (AUC={auc:.2f})',
                 color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)])
        
        print(f"{dataset_name} {model_name}: AUC={auc:.4f}, ACC={acc:.4f}, F1={f1:.4f}")
        
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curves for {dataset_name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
   
    plt.savefig(f"{dataset_name}_roc_curves.png", dpi=300)
    plt.show()
    
    return model_metrics


# In[7]:


all_metrics = {}

for name, dataset in datasets.items():
    dataset.index = dataset.index.astype(str).str.strip().str.lower()
    labels.index = labels.index.astype(str).str.strip().str.lower()
    
    common_indices = extract_indices(dataset, labels)  
    y = labels.loc[common_indices].dropna()    
    X_filtered = dataset.loc[y.index]    
    if not X_filtered.empty and not y.empty:
        metrics = train_and_evaluate(X_filtered, y, name)
        all_metrics[name] = metrics
    else:
        print(f"Skipping {name} due to empty dataset after filtering.")
print("\nFinal Results:")
for dataset_name, metrics in all_metrics.items():
    print(f"\n{dataset_name} 结果:")
    print("{:<20} {:<10} {:<10} {:<10}".format('模型', 'AUC', 'ACC', 'F1'))
    for model_name, scores in metrics.items():
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            model_name, 
            scores['AUC'],
            scores['ACC'],
            scores['F1']
        ))


# In[ ]:




