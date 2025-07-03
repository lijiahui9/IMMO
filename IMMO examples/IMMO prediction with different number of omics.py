#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression  # 导入 LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import os
os.chdir(r"C:\Users\LJH\Desktop\论文\1")
import tensorflow as tf
from tensorflow import keras
import sys
import scipy as sp
import scipy.sparse
import h5py
#import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import History
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
import pandas as pd
import numpy as np


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

# 非NA的位置设置为1，NA的位置设置为0
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


# In[3]:


file_path = r"C:\Users\LJH\Desktop\论文\1\ID_processed.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
cell_types = df['diagnosis'].astype(str)
cell_ids=df.index
label =pd.read_excel("ID_processed.xlsx", header=None)


# In[4]:


all_rows = set(x.index).union(set(y.index)).union(set(z.index))
x_re0 = x.reindex(all_rows, fill_value=np.nan)
y_re0 = y.reindex(all_rows, fill_value=np.nan)
z_re0 = z.reindex(all_rows, fill_value=np.nan)

def extract_indices(x, y, z=None):
    if z is None:
        return x.index.intersection(y.index)
    else:
        return x.index.intersection(y.index).intersection(z.index)

common_indices = extract_indices(x, y, z)
xy_common_indices = extract_indices(x, y)
yz_common_indices = extract_indices(y, z)
xz_common_indices = extract_indices(x, z)
all_rows = set(x.index).union(set(y.index)).union(set(z.index))
xy_all_indices = x.index.union(y.index)
yz_all_indices = y.index.union(z.index)
xz_all_indices = x.index.union(z.index)
all_rows_list = list(all_rows)


# In[9]:


from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve

X = pd.read_csv(r"C:\Users\LJH\Desktop\论文\1\latent_representation_110_dynamic.csv", index_col=1)
X = pd.DataFrame(X)
X.set_index('samples', inplace=True)
IBD_common = X.loc[common_indices]
bol_tra_common = X.loc[xy_common_indices]
tra_bio_common = X.loc[yz_common_indices]
bol_bio_common = X.loc[xz_common_indices]
tra_bol_all = X.loc[xy_all_indices]
tra_bio_all = X.loc[yz_all_indices]
bol_bio_all = X.loc[xz_all_indices]
IBD_all = X.loc[all_rows_list]
label = pd.read_excel("ID_processed.xlsx", header=None)
def get_labels(X, label):
    y0 = np.ones(X.shape[0])
    for i in range(X.shape[0]):
        index = label[label[0] == X.index[i].upper()].index
        if label.iloc[index, 1].values[0] == "nonIBD":
            y0[i] = 0
    return y0
def train_evaluate_models(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models = {
        'SVM': SVC(probability=True),
        'GBDT': GradientBoostingClassifier(),
        'Neural Network': MLPClassifier(max_iter=500),
        'LDA': LinearDiscriminantAnalysis(),
        'Ridge Regression': RidgeClassifierCV(),
        'RF': RandomForestClassifier()
    }
    
    model_aucs = {}
    model_accs = {}
    model_f1_scores = {}
    plt.figure(figsize=(12, 10))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 1))]
        

    model_aucs = {}
    model_accs = {}
    model_f1_scores = {}
    plt.figure(figsize=(12, 10))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 1))]
    
    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"): 
            scores = model.decision_function(X_test)
            probas = (scores - scores.min()) / (scores.max() - scores.min())
        else: 
            probas = model.predict(X_test)
        
        # 计算 AUC
        auc = roc_auc_score(y_test, probas)
        fpr, tpr, _ = roc_curve(y_test, probas)
        plt.plot(fpr, tpr, label=f'{dataset_name} - {name} (AUC = {auc:.4f})',
                 color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)])

        y_pred = model.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred), 4)
        f1 = round(f1_score(y_test, y_pred), 4)
        
        print(f"{dataset_name} {name} AUC: {auc:.4f}, ACC: {acc:.4f}, F1-Score: {f1:.4f}")

        model_aucs[name] = auc
        model_accs[name] = acc
        model_f1_scores[name] = f1

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
    
    return model_aucs, model_accs, model_f1_scores

all_auc_scores = {}
all_acc_scores = {}
all_f1_scores = {}
datasets = {
    'IBD_common': X.loc[common_indices],
    'bol_tra_common': X.loc[xy_common_indices],
    'tra_bio_common': X.loc[yz_common_indices],
    'bol_bio_common': X.loc[xz_common_indices],
    'tra_bol_all': X.loc[xy_all_indices],
    'tra_bio_all': X.loc[yz_all_indices],
    'bol_bio_all': X.loc[xz_all_indices],
    'IBD_all': X.loc[all_rows_list]
}

for dataset_name, data in datasets.items():
    y = get_labels(data, label)
    aucs, accs, f1_scores = train_evaluate_models(data, y, dataset_name)
    all_auc_scores[dataset_name] = aucs
    all_acc_scores[dataset_name] = accs
    all_f1_scores[dataset_name] = f1_scores

for dataset_name in datasets.keys():
    print(f"\n{dataset_name}:")
    for model_name in all_auc_scores[dataset_name].keys():
        print(f"  {model_name} AUC: {all_auc_scores[dataset_name][model_name]:.4f}, "
              f"ACC: {all_acc_scores[dataset_name][model_name]:.4f}, "
              f"F1-Score: {all_f1_scores[dataset_name][model_name]:.4f}")


# In[ ]:




