'''
Description: 
Author: Rocky Hoo
Date: 2021-05-11 10:20:23
LastEditTime: 2021-05-12 17:03:40
LastEditors: Please set LastEditors
CopyRight: 
Copyright (c) 2021 XiaoPeng Studio
'''
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
sns.set(style='ticks')
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
#%%
train_data=pd.read_csv("zhengqi_train.txt",sep="\t")
test_data=pd.read_csv("zhengqi_test.txt",sep="\t")
#%%
train_data.head()
test_data.head()
# %%
''' 箱型图异常值分析 '''
plt.figure(figsize=(18,10))
plt.boxplot(x=train_data.values,labels=train_data.columns)
plt.hlines([-7.5,7.5],0,40,colors="r")
plt.show()
# %%
''' 去除异常值 '''
train_data=train_data[train_data['V9']>-7.5]
test_data=test_data[test_data['V9']>-7.5]
display(train_data.describe())
display(test_data.describe())
#%%
# %%
''' 最大值和最小值归一化 '''
features_cols=[col for col in train_data.columns if col not in ["target"]]
train_data_features=train_data[features_cols]
test_data_features=test_data[features_cols]
min_max_model=MinMaxScaler()
train_data_scalar=min_max_model.fit(train_data_features)
train_data_scalar=min_max_model.transform(train_data_features)

test_data_scalar=min_max_model.fit(test_data_features)
test_data_scalar=min_max_model.transform(test_data_features)

test_data=pd.DataFrame(test_data_scalar)
train_data_scalar=pd.DataFrame(train_data_scalar)
train_data_scalar["target"]=train_data["target"]
train_data=train_data_scalar
display(train_data.describe())
display(test_data.describe())
# %%
''' KDE分析'''
''' 对比变量在训练集和测试集中的分布情况 '''
'''通过KDE分析，删除分布差异较大特征V5 V17 V21 V22 V27 V35 V36 '''
plt.figure(figsize=(8,4),dpi=150)
ax=sns.kdeplot(train_data[0],color="r",shade=True)
ax=sns.kdeplot(test_data[0],color="b",shade=True)
ax.set_xlabel('V0')
ax.set_ylabel("Frequency")
ax.legend(["train","test"])
#%%
# 所有变量的KDE分析
dist_rows=len(test_data.columns)
dist_cols=6
each_width=4
plt.figure(figsize=(each_width*dist_cols,each_width*dist_rows),dpi=150)
pic_num=1
for col in range(dist_rows):
    plt.subplot(dist_rows,dist_cols,pic_num)
    ax=sns.kdeplot(train_data[col],color="r",shade=True)
    ax=sns.kdeplot(test_data[col],color="b",shade=True)
    ax.set_xlabel("V"+str(col))
    ax.set_ylabel("Frequency")
    ax.legend(["train","test"])
    pic_num+=1
plt.show()
#%%
len(train_data.columns)
# %%
''' 热力图绘制,相关性分析 '''
plt.figure(figsize=(20,20))
cols=train_data.columns.tolist()
mcorr=train_data[cols].corr(method="spearman")
mask=np.zeros_like(mcorr,dtype=np.bool)
#将上三角设置为True 则只显示下三角矩阵的数据
mask[np.triu_indices_from(mcorr)]=True
cmap=sns.diverging_palette(220,10,as_cmap=True)
g=sns.heatmap(mcorr,mask=mask,cmap=cmap,square=True,annot=True,fmt="0.2f")
mcorr=mcorr.abs()
filter_mcorr=mcorr[mcorr["target"]>0.1]["target"]
print(filter_mcorr.sort_values(ascending=False))
# %%
''' 多重共线性分析 '''
# VIF:共线性方差膨胀系数，衡量共线性程度。值越大则共线性程度越大
filter_cols=[5,17,21,22,27,35,36]
# 筛选出需要PCA处理的特征
final_cols=[col for col in filter_mcorr.index[:-1] if col not in filter_cols]
X=np.matrix(train_data[final_cols])
VIF_list=[variance_inflation_factor(X,i) for i in range(X.shape[1])]
# %%
'''  PCA降维，共线性转换'''
pca=PCA(n_components=16)
pca_new_train_data=pca.fit_transform(train_data[final_cols])
pca_new_test_data=pca.fit_transform(test_data)
pca_new_train_data=pd.DataFrame(pca_new_train_data)
pca_new_test_data=pd.DataFrame(pca_new_test_data)
pca_new_train_data["target"]=train_data["target"]
display(pca_new_train_data.describe())
display(train_data.describe())
# %%
