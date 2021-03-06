'''
Description: 增加Stacking预测
Author: Rocky Hoo
Date: 2021-06-03 10:25:54
LastEditTime: 2021-06-09 21:48:45
LastEditors: Please set LastEditors
CopyRight: 
Copyright (c) 2021 XiaoPeng Studio
'''
'''
Description: 训练模型
Author: Rocky Hoo
Date: 2021-05-25 13:15:52
LastEditTime: 2021-06-02 23:29:38
LastEditors: Please set LastEditors
CopyRight: 
Copyright (c) 2021 XiaoPeng Studio
'''
#%%
from seaborn.axisgrid import Grid
from sklearn.model_selection import train_test_split #训练集测试集切分函数
from sklearn.metrics import mean_squared_error #均方差
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures #多项式特征构造
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
import Learning_Curve
import matplotlib.pyplot as plt
import lightgbm as lgb 
import pandas as pd
import numpy as np
import re
#%%
# 读取数据
new_train_pca_16=pd.read_csv("./data/new_train_pca_16_origin.txt")
new_train_pca_16 = new_train_pca_16.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

new_train=pd.read_csv("./data/new_train.txt")
new_train = new_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

new_train_pca_16=new_train_pca_16.fillna(0)
new_train=new_train.fillna(0)

linear_train=new_train_pca_16[new_train_pca_16.columns[:-1]]
train=new_train[new_train.columns[:-1]]

target=new_train["target"]
target_linear=new_train_pca_16["target"]

models=[]
linear_models=[]
# 处理特殊字符
#%%
train_data,test_data,train_target,test_target=train_test_split(train,target,test_size=0.3,random_state=0)
linear_train_data,linear_test_data,linear_train_target,linear_test_target=train_test_split(linear_train,target_linear,test_size=0.3,random_state=0)

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
    KNN_Model=KNeighborsRegressor()
    n_neighbors=np.arange(3,11,1)
    param_grid={"n_neighbors":n_neighbors}
    clf=RandomizedSearchCV(KNN_Model,param_grid,cv=5)
    clf.fit(train_data,train_target)
    test_pred6=clf.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred6)
    print("KNNRegression score:%f"%score_test)
    return clf,test_pred6
    
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
    randomForestRegressor=RandomForestRegressor()
    parameters_grid={"n_estimators":[50,100,200,300],"max_depth":[1,2,3]}
    clf=RandomizedSearchCV(randomForestRegressor,parameters_grid,cv=5)
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
    clf=SGDRegressor(max_iter=100000,tol=1e-3)
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
#%%
# 岭回归
def RidgeRegression(train_data,train_target,test_data,test_target):
    Ridge_Model=Ridge()
    alpha_range=np.arange(0.25,6,0.25)
    param_grid={"alpha":alpha_range}
    clf=RandomizedSearchCV(Ridge_Model,param_grid,cv=5)
    clf.fit(train_data,train_target)
    test_pred6=clf.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred6)
    print("RidgeRegression score:%f"%score_test)
    return clf,test_pred6
#%%
# lasso回归
def LassoRegression(train_data,train_target,test_data,test_target):
    Lasso_Model=Lasso()
    alpha_range=np.arange(1e-4,1e-3,4e-5)
    param_grid={"alpha":alpha_range}
    clf=RandomizedSearchCV(Lasso_Model,param_grid,cv=5)
    clf.fit(train_data,train_target)
    test_pred7=clf.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred7)
    print("LassoRegression score:%f"%score_test)
    return clf,test_pred7
#%%
# elasticNet回归
def ElasticNetRegression(train_data,train_target,test_data,test_target):
    ElasticNet_Model=ElasticNet()
    alpha_range=np.arange(1e-4,1e-3,1e-4)
    param_grid={"alpha":alpha_range,"l1_ratio": np.arange(0.1,1.0,0.1)}
    clf=RandomizedSearchCV(ElasticNet_Model,param_grid,cv=5)
    clf.fit(train_data,train_target)
    test_pred8=clf.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred8)
    print("ElasticNetRegression score:%f"%score_test)
    return clf,test_pred8
#%%
# SVR回归(拟合不了?)
def SVRRegression(train_data,train_target,test_data,test_target):
    SVR_Model=LinearSVR()
    c_range=np.arange(0.1,1.0,0.1)
    param_grid={"C":c_range,"max_iter": [10000]}
    clf=RandomizedSearchCV(SVR_Model,param_grid,cv=5)
    clf.fit(train_data,train_target)
    test_pred9=clf.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred9)
    print("SVRRegression score:%f"%score_test)
    return clf,test_pred9
#%%
# 梯度下降树回归
def GBDTRegression(train_data,train_target,test_data,test_target):
    GBDT_Model=GradientBoostingRegressor()
    n_estimators=np.arange(150,400,50)
    max_depth=[1,2,3]
    min_samples_split=[5,6,7]
    param_grid={"n_estimators":n_estimators,"min_samples_split": min_samples_split,"max_depth":max_depth}
    clf=RandomizedSearchCV(GBDT_Model,param_grid,cv=5)
    clf.fit(train_data,train_target)
    test_pred10=clf.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred10)
    print("GBDTRegression score:%f"%score_test)
    return clf,test_pred10
#%%
# XGB模型回归
def XGBRegression(train_data,train_target,test_data,test_target):
    XGB_Model=XGBRegressor(booster='gbtree',colsample_bytree=0.8, gamma=0.1, 
                                 learning_rate=0.02, 
                                 min_child_weight=0.8,
                                 reg_alpha=0, reg_lambda=1,
                                 subsample=0.8, silent=1,
                                 nthread = 2)
    n_estimators=np.arange(100,600,100)
    max_depth=[1,2,3,5]
    param_grid={"n_estimators":n_estimators,"max_depth":max_depth}
    clf=RandomizedSearchCV(XGB_Model,param_grid,cv=5)
    clf.fit(train_data,train_target)
    test_pred11=clf.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred11)
    print("XGBRegression score:%f"%score_test)
    return clf,test_pred11
#%%
# MLP回归
def MLPRegression(train_data,train_target,test_data,test_target):
    MLP_Model=MLPRegressor()
    hidden_layer_sizes=[(100,100,),(128,32,8,)]
    activation=["logistic","relu","identity"]
    max_iter=[1000,1500,2000]
    param_grid={"hidden_layer_sizes":hidden_layer_sizes,"activation":activation,"max_iter":max_iter}
    clf=GridSearchCV(MLP_Model,param_grid,cv=5)
    clf.fit(train_data,train_target)
    test_pred12=clf.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred12)
    print("MLPRegression score:%f"%score_test)
    return clf,test_pred12
#%%
""" (多项式)核脊回归 """
def KRRRegression1(train_data,train_target,test_data,test_target):
    KRR_Model1=KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    KRR_Model1.fit(train_data,train_target)
    test_pred13=KRR_Model1.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred13)
    print("KRRRegression1 score:%f"%score_test)
    return KRR_Model1,test_pred13
#%%
""" (线性)核脊回归 """
def KRRRegression2(train_data,train_target,test_data,test_target):
    KRR_Model2=KernelRidge(alpha=1.5, kernel='linear', degree=2, coef0=2.5)  
    KRR_Model2.fit(train_data,train_target)
    test_pred14=KRR_Model2.predict(test_data)
    score_test=mean_squared_error(test_target,test_pred14)
    print("KRRRegression2 score:%f"%score_test)
    return KRR_Model2,test_pred14
#%%
#Average模型
def Linear_Average(test_data):
    predict_total=np.zeros((test_data.shape[0],))
    for model in linear_models:
        predict_total+=model.predict(test_data)
    return pd.Series(np.round(predict_total/len(linear_models),3))

def Average(test_data):
    predict_total=np.zeros((test_data.shape[0],))
    for model in models:
        predict_total+=model.predict(test_data)
    return pd.Series(np.round(predict_total/len(models),3))
#%%
""" 创建stack特征 """
def Create_Stack_Features(data,models):
    new_train={}
    columns=[]
    for model in models:
        columns.append(model)
        new_train[model]=model.predict(data)   
    return pd.DataFrame(new_train,columns=columns)
#%%
'''
@description: 
@param  {*}
@return {*}返回stack处理后的测试集数据
@param {*} train:训练集所有数据
@param {*} test:线上测试集所有数据
'''
def Linear_Stacking(train,test,pred):
    stacked_train=Create_Stack_Features(train,linear_models)
    stacked_pred=Create_Stack_Features(pred,linear_models)
    stacked_test=Create_Stack_Features(test,linear_models)

    linear_models.clear()
    Linear_K_Fold(stacked_train)
    return pd.DataFrame(stacked_test),pd.DataFrame(stacked_pred)
    
def Stacking(train,test,pred):
    stacked_train=Create_Stack_Features(train,models)
    stacked_pred=Create_Stack_Features(pred,models)
    stacked_test=Create_Stack_Features(test,models)

    models.clear()
    K_Fold(stacked_train)
    return pd.DataFrame(stacked_test),pd.DataFrame(stacked_pred)
    
#%%
def Linear_K_Fold(train):
    kf=KFold(n_splits=5,random_state=1,shuffle=True)
    for k,(train_index,test_index) in enumerate(kf.split(train)):
        temp_model=[]
        train_data=train.values[train_index]
        test_data=train.values[test_index]
        train_target=target_linear[train_index]
        test_target=target_linear[test_index]
        model1,test_pred1=MyLinearRegression(train_data,train_target,test_data,test_target)
        # model2,test_pred2=RFRegression(train_data,train_target,test_data,test_target)
        # model3,test_pred3=LGBRegression(train_data,train_target,test_data,test_target)
        # model4,test_pred4=KNNRegression(train_data[0],train_target,test_data,test_target)
        # model5,test_pred5=SGDRegression(train_data[0],train_target,test_data,test_target)
        model6,test_pred6=RidgeRegression(train_data,train_target,test_data,test_target)
        model7,test_pred7=LassoRegression(train_data,train_target,test_data,test_target)
        model8,test_pred8=ElasticNetRegression(train_data,train_target,test_data,test_target)
        model9,test_pred9=SVRRegression(train_data,train_target,test_data,test_target)
        # model10,test_pred10=GBDTRegression(train_data,train_target,test_data,test_target)
        # model11,test_pred11=XGBRegression(train_data,train_target,test_data,test_target)
        # model12,test_pred12=MLPRegression(train_data,train_target,test_data,test_target)
        model13,test_pred13=KRRRegression1(train_data,train_target,test_data,test_target)
        model14,test_pred14=KRRRegression2(train_data,train_target,test_data,test_target)
        linear_models.append(model1)
        # models.append(model2)
        # models.append(model3)
        # models.append(model4)
        # models.append(model5)
        linear_models.append(model6)
        linear_models.append(model7)
        linear_models.append(model8)
        linear_models.append(model9)
        # models.append(model10)
        # models.append(model11)
        # models.append(model12)
        linear_models.append(model13)
        linear_models.append(model14)
        
        temp_model.append(test_pred1)
        # temp_model.append(test_pred2)
        # temp_model.append(test_pred3)
        # models.append(test_pred4)
        # models.append(model5)
        temp_model.append(test_pred6)
        temp_model.append(test_pred7)
        temp_model.append(test_pred8)
        temp_model.append(test_pred9)
        # temp_model.append(test_pred10)
        # temp_model.append(test_pred11)
        # temp_model.append(test_pred12)
        temp_model.append(test_pred13)
        temp_model.append(test_pred14)

        score_test=mean_squared_error(test_target,sum(temp_model)/len(temp_model))
        print(k," fold","train_mse:",score_test,'\n')
#%%
def K_Fold(train):
    kf=KFold(n_splits=5,random_state=1,shuffle=True)
    for k,(train_index,test_index) in enumerate(kf.split(train)):
        temp_model=[]
        train_data=train.values[train_index]
        test_data=train.values[test_index]
        train_target=target[train_index]
        test_target=target[test_index]
        # model1,test_pred1=MyLinearRegression(train_data,train_target,test_data,test_target)
        model2,test_pred2=RFRegression(train_data,train_target,test_data,test_target)
        model3,test_pred3=LGBRegression(train_data,train_target,test_data,test_target)
        # model4,test_pred4=KNNRegression(train_data[0],train_target,test_data,test_target)
        # model5,test_pred5=SGDRegression(train_data[0],train_target,test_data,test_target)
        # model6,test_pred6=RidgeRegression(train_data,train_target,test_data,test_target)
        # model7,test_pred7=LassoRegression(train_data,train_target,test_data,test_target)
        # model8,test_pred8=ElasticNetRegression(train_data,train_target,test_data,test_target)
        # model9,test_pred9=SVRRegression(train_data,train_target,test_data,test_target)
        model10,test_pred10=GBDTRegression(train_data,train_target,test_data,test_target)
        model11,test_pred11=XGBRegression(train_data,train_target,test_data,test_target)
        model12,test_pred12=MLPRegression(train_data,train_target,test_data,test_target)
        # model13,test_pred13=KRRRegression1(train_data,train_target,test_data,test_target)
        # model14,test_pred14=KRRRegression2(train_data,train_target,test_data,test_target)
        # models.append(model1)
        models.append(model2)
        models.append(model3)
        # models.append(model4)
        # models.append(model5)
        # models.append(model6)
        # models.append(model7)
        # models.append(model8)
        # models.append(model9)
        models.append(model10)
        models.append(model11)
        models.append(model12)
        # models.append(model13)
        # models.append(model14)
        
        # temp_model.append(test_pred1)
        temp_model.append(test_pred2)
        temp_model.append(test_pred3)
        # temp_model.append(test_pred4)
        # temp_model.append(test_pred5)
        # temp_model.append(test_pred6)
        # temp_model.append(test_pred7)
        # temp_model.append(test_pred8)
        # temp_model.append(test_pred9)
        temp_model.append(test_pred10)
        temp_model.append(test_pred11)
        temp_model.append(test_pred12)
        # temp_model.append(test_pred13)
        # temp_model.append(test_pred14)

        score_test=mean_squared_error(test_target,sum(temp_model)/len(temp_model))
        print(k," fold","train_mse:",score_test,'\n')
#%%
final_test_data=pd.read_csv("./data/new_test.txt")
# %%
# K_Fold(train_data)
Linear_K_Fold(linear_train)
K_Fold(train)
stacked_test_linear,stacked_pred_linear=Linear_Stacking(linear_train_data,linear_test_data,final_test_data)
stacked_test,stacked_pred=Stacking(train_data,test_data,final_test_data)
ans2=Linear_Average(stacked_pred_linear)
ans1=Average(stacked_pred)
ans=pd.Series(np.round(ans1+ans2/2,3))
# ans=Average(final_test_data)
# %%
ans.to_csv('./predict_.txt',header = None,index = False)
# %%
train_ans=Average(stacked_test)
linear_train_ans=Linear_Average(stacked_test_linear)
# train_ans=pd.Series(np.round(Average(stacked_test)+Linear_Average(stacked_test_linear)/2,3))
# train_ans=Average(test_data)
print("stack_test score:",mean_squared_error(test_target,train_ans))
print("linear_stack_test score:",mean_squared_error(linear_test_target,linear_train_ans))
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
