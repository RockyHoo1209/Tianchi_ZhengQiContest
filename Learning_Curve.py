'''
Description: 绘制学习曲线和验证曲线
Author: Rocky Hoo
Date: 2021-05-26 17:13:20
LastEditTime: 2021-05-27 10:44:22
LastEditors: Please set LastEditors
CopyRight: 
Copyright (c) 2021 XiaoPeng Studio
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve,validation_curve

plt.figure(figsize=(18,10),dpi=150)

def plot_learning_curve(estimator,title,train_data,target,ylim=None,cv=None,n_job=1,train_sizes=np.linspace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        # y坐标最大值
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores=learning_curve(
        estimator,train_data,target,cv=cv,n_jobs=n_job,train_sizes=train_sizes)
    train_score_mean=np.mean(train_scores,axis=1)
    test_score_mean=np.mean(test_scores,axis=1)
    train_score_std=np.std(train_scores,axis=1)
    test_score_std=np.std(test_scores,axis=1)
    # 显示网格线
    plt.grid()
    plt.fill_between(train_sizes,train_score_mean-train_score_std,train_score_mean+train_score_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_score_mean-test_score_std,test_score_mean+test_score_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_score_mean,'o-',color='r',label="Training Score")
    plt.plot(train_sizes,test_score_mean,'o-',linestyle="--",color='g',label="Cross-Validation Score")
    plt.legend(loc="best")
    return plt


def plot_validating_curve(estimator,title,test_data,test_target,ylim=None,cv=None,n_job=1):
    param_range=[1000,5000,10000,50000]
    plt.figure()
    plt.title(title)
    if ylim is not None:
        # y坐标最大值
        plt.ylim(*ylim)
    plt.xlabel("n_estimators")
    plt.ylabel("Score")
    train_scores,test_scores=validation_curve(
        estimator,test_data,test_target,param_name="n_estimators",param_range=param_range,cv=cv,scoring='r2',n_jobs=n_job)
    train_score_mean=np.mean(train_scores,axis=1)
    test_score_mean=np.mean(test_scores,axis=1)
    train_score_std=np.std(train_scores,axis=1)
    test_score_std=np.std(test_scores,axis=1)
    # 显示网格线
    plt.grid()
    plt.semilogx(param_range,train_score_mean,label="Training Score",color='r')
    plt.fill_between(param_range,train_score_mean-train_score_std,train_score_mean+train_score_std,alpha=0.2,color='r')
    plt.semilogx(param_range,test_score_mean,label="Cross-validation score",color='g')
    plt.fill_between(param_range,test_score_mean-test_score_std,test_score_mean+test_score_std,alpha=0.2,color='g')
    plt.legend(loc="best")
    plt.show()
    return plt