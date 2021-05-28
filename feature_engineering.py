'''
Description: 特征工程
Author: Rocky Hoo
Date: 2021-05-11 10:20:23
LastEditTime: 2021-05-28 23:15:29
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
from scipy import stats
sns.set(style='ticks')
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
#%%
train_data=pd.read_csv("zhengqi_train.txt",sep="\t")
test_data=pd.read_csv("zhengqi_test.txt",sep="\t")
temp_train_data=train_data.copy()
temp_test_data=test_data.copy()
temp_train_data["origin"]="train"
temp_test_data["origin"]="test"
data_all=pd.concat([temp_train_data,temp_test_data],axis=0,ignore_index=True)
#%%
train_data.head()
#%%
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
print(train_data.describe())
print(test_data.describe())
# %%
''' 最大值和最小值归一化 '''
def scala_minmax(train_data,test_data,data_all):
    features_cols=[col for col in train_data.columns if col not in ["target"]]
    train_data_features=train_data[features_cols]
    test_data_features=test_data[features_cols]
    data_all_features=data_all[features_cols]
    # data_all_features=pd.concat([train_data_features,test_data_features],axis=0,ignore_index=True)
    min_max_model=MinMaxScaler()
    # 数据量小，用所有数据归一化处理，保证分布相同
    train_data_scalar=min_max_model.fit(data_all_features)
    train_data_scalar=min_max_model.transform(train_data_features)
    test_data_scalar=min_max_model.transform(test_data_features)

    test_data=pd.DataFrame(test_data_scalar)
    train_data_scalar=pd.DataFrame(train_data_scalar)
    train_data_scalar["target"]=train_data["target"]
    train_data=train_data_scalar
    temp_train_data=train_data.copy()
    temp_test_data=test_data.copy()
    temp_train_data["origin"]="train"
    temp_test_data["origin"]="test"
    data_all=pd.concat([temp_train_data,temp_test_data],axis=0,ignore_index=True)
    return train_data,test_data,data_all
#%%
train_data,test_data,data_all=scala_minmax(train_data,test_data,data_all)
#%%
train_data.head()
print(train_data.describe())
print(test_data.describe())
# %%
''' KDE分析'''
''' 对比变量在训练集和测试集中的分布情况 '''
'''通过KDE分析，删除分布差异较大特征V5 V8 V17 V21 V22 V27 V35 '''
plt.figure(figsize=(8,4),dpi=150)
ax=sns.kdeplot(train_data[0],color="r",shade=True)
ax=sns.kdeplot(test_data[0],ax=ax,color="b",shade=True)
ax.set_xlabel('V0')
ax.set_ylabel("Frequency")
ax.legend(["train","test"])
#%%
# 所有变量的KDE分析
dist_rows=len(train_data.columns[:-2])
train_cols=train_data.columns[:-2]
dist_cols=6
each_width=4
plt.figure(figsize=(each_width*dist_cols,each_width*dist_rows),dpi=150)
pic_num=1
# for col in range(dist_rows):
for col in train_cols:
    try:
        plt.subplot(dist_rows,dist_cols,pic_num)
        ax=sns.kdeplot(train_data[col],color="r",shade=True)
        ax=sns.kdeplot(test_data[col],color="b",shade=True)
        ax.set_xlabel("V"+str(col))
        ax.set_ylabel("Frequency")
        ax.legend(["train","test"])
        pic_num+=1
    except:
        continue
plt.show()
#%%
# 直方图和KDE图对比
# 结合上面两种方法筛选特征
fig = plt.figure(figsize=(10, 10))
for i in range(len(train_data.columns)-2):
    g = sns.FacetGrid(data_all,col="origin")
    g = g.map(sns.distplot, data_all.columns[i])
#%%
''' 绘制回归图、Q-Q图(查看数据分布是否近似于正态分布)，进一步筛选特征 '''
figure_cols=3
figure_rows=len(train_data.columns)
plt.figure(figsize=(5*figure_cols,4*figure_rows))
i=0
for col in train_data.columns:
    i+=1
    ax=plt.subplot(figure_rows,figure_cols,i)
    sns.regplot(x=train_data[col],y="target",data=train_data,ax=ax,
    scatter_kws={"marker":".","s":3,"alpha":0.3},line_kws={"color":"k"})
    plt.xlabel(col)
    plt.ylabel('target')
    i+=1
    ax=plt.subplot(figure_rows,figure_cols,i)
    # kde图+直方图+正太曲线图
    sns.distplot(train_data[col].dropna(),fit=stats.norm)
    i+=1
    ax=plt.subplot(figure_rows,figure_cols,i)
    res=stats.probplot(train_data[col],plot=plt)
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
g=sns.heatmap(mcorr,mask=mask,cmap=cmap,square=True,annot=True,fmt="0.2f",linewidths=1)
mcorr=mcorr.abs()
filter_mcorr=mcorr[mcorr["target"]<0.1]["target"]
print(filter_mcorr.sort_values(ascending=False))
#%%
'''根据分析删除部分特征  '''
#根据KDE删除特征
drop_col=[5, 8, 17, 21, 22, 27, 35]
#根据热力图，去掉一些相关性较弱的特征
# drop_col=[col for col in drop_col1 if col not in filter_mcorr]
drop_col+=filter_mcorr.index.tolist()
drop_col=list(set(drop_col))
train_data.drop(drop_col,axis=1,inplace=True)
test_data.drop(drop_col,axis=1,inplace=True)
# %%
# 处理nan和inf竖直
train_data[np.isnan(train_data)]=0
train_data[np.isinf(train_data)]=0
test_data[np.isinf(train_data)]=0
test_data[np.isinf(train_data)]=0
train_data.head()
#%%
# 数值归一化
def min_max(cols):
    return (cols-cols.min())/(cols.max()-cols.min())
#%%
''' Box-cox(正态化)变换 '''
''' 线性回归基于正态分布，故需将数据正态化 '''
cols_numeric=list(train_data.columns[:-2])
fcols=6
frows=len(cols_numeric)
plt.figure(figsize=(4*fcols,4*frows))
i=0
for col in cols_numeric:
    i+=1
    dat=train_data[[col,"target"]]
    plt.subplot(frows,fcols,i)
    sns.distplot(dat[col],fit=stats.norm)
    plt.title(str(col)+' Original')
    plt.xlabel('')
    
    i+=1
    plt.subplot(frows,fcols,i)
    _=stats.probplot(dat[col],plot=plt)
    # skew意思为偏差
    plt.title("skew=%.4f"%(stats.skew(dat[col])))
    plt.xlabel('')
    plt.ylabel('')
    
    i+=1
    plt.subplot(frows,fcols,i)
    # 连线
    plt.plot(dat[col],dat["target"],'.',alpha=0.5)
    # 取相关矩阵中的(0,1)进行显示
    plt.title("corr=%.2f"%(np.corrcoef(dat[col],dat["target"])[0][1]))
    
    i+=1
    plt.subplot(frows,fcols,i)
    # 归一化后防止dat[col]出现负数
    trains_var,lambda_var=stats.boxcox(dat[col].dropna()+1)
    trains_var=min_max(trains_var)
    sns.distplot(trains_var,fit=stats.norm)
    plt.title(str(col)+"Tramsformed")
    plt.xlabel('')
    
    i+=1
    plt.subplot(frows,fcols,i)
    plt.plot(trains_var,dat["target"],".",alpha=0.5)
    plt.title("corr=%.2f"%(np.corrcoef(trains_var,dat["target"])[0][1]))
#%%
''' 对所有列进行boxcox转换 '''
trans_cols=train_data.columns[:-2]
for col in trans_cols:
    train_data.loc[:,col], _ = stats.boxcox(train_data.loc[:,col]+1)
    test_data.loc[:,col], _ = stats.boxcox(test_data.loc[:,col]+1)
    data_all.loc[:,col],_=stats.boxcox(data_all.loc[:,col]+1)
    data_all.loc[:,col]=min_max(data_all.loc[:,col])
# 归一化
train_data,test_data,data_all=scala_minmax(train_data,test_data,data_all)
#%%
# 查看target是否满足正态分布
print(data_all.target.describe())
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.distplot(data_all.target.dropna() , fit=stats.norm);
plt.subplot(1,2,2)
_=stats.probplot(data_all.target.dropna(), plot=plt)
#%%
#Log Transform SalePrice to improve normality
# Q:why this?
sp = train_data.target
train_data.target =np.power(1.5,sp)
print(train_data.target.describe())

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.distplot(train_data.target1.dropna(),fit=stats.norm);
plt.subplot(1,2,2)
_=stats.probplot(train_data.target1.dropna(), plot=plt)
#%%
''' 多重共线性分析 '''
# VIF:共线性方差膨胀系数，衡量共线性程度。值越大则共线性程度越大
X=np.matrix(train_data)
VIF_list=[variance_inflation_factor(X,i) for i in range(X.shape[1])]
#%%
VIF_list
# %%
'''  PCA降维，共线性转换'''
pca=PCA(n_components=16)
pca_new_train_data=pca.fit_transform(train_data)
pca_new_test_data=pca.fit_transform(test_data)
pca_new_train_data=pd.DataFrame(pca_new_train_data)
pca_new_test_data=pd.DataFrame(pca_new_test_data)
pca_new_train_data["target"]=train_data["target"]
print(pca_new_train_data.describe())
print(train_data.describe())
# %%
pca_new_train_data.to_csv("./new_train_pca_16.txt")
# %%
pca_new_test_data.to_csv("./new_test_pca_16.txt")

# %%
print("ok!")
# %%
