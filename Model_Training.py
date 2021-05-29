'''
Description: 训练模型
Author: Rocky Hoo
Date: 2021-05-25 13:15:52
LastEditTime: 2021-05-29 21:40:51
LastEditors: Please set LastEditors
CopyRight: 
Copyright (c) 2021 XiaoPeng Studio
'''
#%%
from sklearn.model_selection import train_test_split #训练集测试集切分函数
from sklearn.metrics import mean_squared_error #均方差
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures #多项式特征构造
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
import Learning_Curve
import matplotlib.pyplot as plt
import lightgbm as lgb 
import pandas as pd
import numpy as np
import re
#%%
# 读取数据
new_train_pca_16=pd.read_csv("./new_train_pca_16.txt")
new_train_pca_16 = new_train_pca_16.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
new_train_pca_16=new_train_pca_16.fillna(0)
train=new_train_pca_16[new_train_pca_16.columns[:-1]]
target=new_train_pca_16["target"]
models=[]
# 处理特殊字符
#%%
train_data,test_data,train_target,test_target=train_test_split(train,target,test_size=0.3,random_state=0)
#%%
# 线性回归 score:0.000
def MyLinearRegression(train_data,train_target,test_data,test_target):
    clf=LinearRegression()
    print(train_data.ndim)
    clf.fit(train_data,train_target)
    # models.append(clf)
    test_pred1=clf.predict(test_data)
    score=mean_squared_error(test_target,test_pred1)
    print("LR SCORE:%f"%score)
    return clf,test_pred1

# %%
def KNNRegression(train_data,train_target,test_data,test_target):
    #  KNN回归 score:0.334
    clf=KNeighborsRegressor(n_neighbors=3)
    # models.append(clf)
    clf.fit(train_data,train_target)
    test_pred2=clf.predict(test_data)
    score=mean_squared_error(test_target,test_pred2)
    print("KNN SCORE:%f"%score)
    return clf,test_pred2
#%%
# 决策树 score:0.000102
clf=DecisionTreeRegressor()
# models.append(clf)
clf.fit(train_data,train_target)
test_pred3=clf.predict(test_data)
score=mean_squared_error(test_target,test_pred3)
print("DTR SCORE:%f"%score)
# %%
# 随机森林 score:0.000023
def RFRegression(train_data,train_target,test_data,test_target):
    clf=RandomForestRegressor(n_estimators=200)
    clf.fit(train_data,train_target)
    # models.append(clf)
    test_pred4=clf.predict(test_data)
    score=mean_squared_error(test_target,test_pred4)
    print("RF SCORE:%f"%score)
    return clf,test_pred4
# %%
def LGBRegression(train_data,train_target,test_data,test_target):
    # LGB算法
    clf=lgb.LGBMRegressor(
        learning_rate=0.01,
        max_depth=-1,
        n_estimators=5000,
        boosting_type="gbdt",
        random_state=2019,
        objective="regression",
    )
    # models.append(clf)
    clf.fit(X=train_data,y=train_target,eval_metric="MSE",verbose=50)
    test_pred5=clf.predict(test_data)
    score=mean_squared_error(test_target,test_pred5)
    print("LGB SCORE:%f"%score)
    return clf,test_pred5
# %%
def PolyRegression():
    # 多项式拟合(拟合失败)
    poly=PolynomialFeatures(degree= 5)
    train_data_poly=poly.fit_transform(train_data)
    test_data_poly=poly.transform(test_data)
    clf=SGDRegressor(max_iter=1000,tol=1e-3)
    # models.append(clf)
    clf.fit(train_data_poly,train_target)
    train_pred=clf.predict(train_data_poly)
    train_score=mean_squared_error(train_target,train_pred)
    test_pred6=clf.predict(test_data_poly)
    score=mean_squared_error(test_target,test_pred6)
    print("Poly3 SCORE:%f"%train_score)
#%%
def SGDRegression(train_data,train_target,test_data,test_target):
    # 随机梯度下降
    clf=SGDRegressor(max_iter=1000,tol=1e-3,penalty="elasticnet",l1_ratio=0.9,alpha=1e-5)
    clf.fit(train_data,train_target)
    test_pred7=clf.predict(test_data)
    score=mean_squared_error(test_target,test_pred7)
    print("SGD SCORE:%f"%score)
    return clf,test_pred7
# %%
#stack模型
def Stack(test_data):
    predict_total=np.zeros((test_data.shape[0],))
    for model in models:
        predict_total+=model.predict(test_data)
    return pd.Series(predict_total/len(models))
#%%
def K_Fold():
    kf=KFold(n_splits=5,random_state=1,shuffle=True)
    for k,(train_index,test_index) in enumerate(kf.split(train)):
        train_data=train.values[train_index],
        test_data=train.values[test_index]
        train_target=target[train_index]
        test_target=target[test_index]
        model1,test_pred1=MyLinearRegression(train_data[0],train_target,test_data,test_target)
        model2,test_pred2=RFRegression(train_data[0],train_target,test_data,test_target)
        model3,test_pred3=LGBRegression(train_data[0],train_target,test_data,test_target)
        # model4,test_pred4=KNNRegression(train_data[0],train_target,test_data,test_target)
        # model5,test_pred5=SGDRegression(train_data[0],train_target,test_data,test_target)
        models.append(model1)
        models.append(model2)
        models.append(model3)
        # models.append(model4)
        # models.append(model5)
        score_test=mean_squared_error(test_target,(test_pred1+test_pred2+test_pred3)/3)
        print(k," fold","train_mse:",score_test,'\n')
#%%
final_test_data=pd.read_csv("./new_test_pca_16.txt")
# %%
K_Fold()
ans=Stack(final_test_data)
# %%
ans.to_csv('./predict_.txt',header = None,index = False)
# %%
train_ans=Stack(test_data)
print("stack_test score:",mean_squared_error(test_target,train_ans))
#%%
# from sklearn.model_selection import ShuffleSplit
# clf=SGDRegressor()
# # clf=SGDRegressor(max_iter=1000,tol=1e-3,penalty="elasticnet",l1_ratio=0.9,alpha=1e-5)
# clf=lgb.LGBMRegressor(
#     learning_rate=0.01,
#     max_depth=-1,
#     # n_estimators=5000,
#     boosting_type="gbdt",
#     random_state=2019,
#     objective="regression",
# )
# clf=RandomForestRegressor()
# title="RandomForestRegressor"
# cv=ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)
# #%%
# Learning_Curve.plot_learning_curve(clf,title,train_data,train_target,cv=cv)
# # %%
# Learning_Curve.plot_validating_curve(clf,title,test_data,test_target,cv=cv)
# %%
