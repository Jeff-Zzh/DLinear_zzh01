import numpy as np
'''
定义了评价回归模型精确度的指标函数
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
    Correlation 皮尔逊相关系数
    相关系数，衡量预测值与真实值之间的线性相关性
    计算两个数组（预测值和真实值）之间的相关系数。相关系数是衡量两个变量之间线性关系强度的一种统计量，其取值范围在-1到1之间
    :param pred:
    :param true:
    :return:
    '''
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0) # 真实值和预测值与它们各自的均值的差值的乘积
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0)) # 每个元素平方的差值乘积之和
    d += 1e-12 # 避免分母为零的情况
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
