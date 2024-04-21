import numpy as np
'''
定义了评价回归模型性能的指标函数
'''

def RSE(pred, true):
    '''
    Relative Squared Error
    相对平方误差，衡量预测值与真实值之间的相对误差
    预测值与真实值之间的平方差除以真实值与其均值之间的平方差。
    :param pred:
    :param true:
    :return:
    '''
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    '''
    Correlation
    相关系数，衡量预测值与真实值之间的线性相关性
    :param pred:
    :param true:
    :return:
    '''
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)

def MAE(pred, true):
    '''
    Mean Absolute Error
    平均绝对误差，衡量预测值与真实值之间的平均绝对差距
    :param pred:
    :param true:
    :return:
    '''
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    '''
    Mean Squared Error
    均方误差，衡量预测值与真实值之间的平均平方差距
    :param pred:
    :param true:
    :return:
    '''
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    '''
    Root Mean Squared Error
    均方根误差，是均方误差的平方根，与 MSE 相对应
    :param pred:
    :param true:
    :return:
    '''
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    '''
    Mean Absolute Percentage Error
    平均绝对百分比误差，衡量预测值与真实值之间的平均绝对百分比差距
    :param pred:
    :param true:
    :return:
    '''
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    '''
    Mean Squared Percentage Error
    平均平方百分比误差，衡量预测值与真实值之间的平均平方百分比差距
    :param pred:
    :param true:
    :return:
    '''
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    '''
    计算预测值和真实值之间的上述所有误差
    :param pred:
    :param true:
    :return:
    '''
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
