import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler # 数据标准化
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    '''
    自定义数据集类，加载数据集，并根据需要进行预处理，例如标准化、时间编码
    '''
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'): # default set for ETTh1.csv数据集，对应ETTh1.csv中内容
        '''

        root_path: 数据集所在的根目录。
        flag: 数据集的类型，可以是 'train'、'test' 或 'val'。
        size: 数据的时间窗口大小，包括seq_len（序列长度）、label_len（标签长度）和pred_len（预测长度）。
        features: 用于指定数据集中包含的特征。默认为 'S'。
        data_path: 数据文件的路径，默认为 'ETTh1.csv'。
        target: 指定时间序列数据中哪个变量是模型要预测的目标，默认为 'OT'。
        scale: 是否对数据进行标准化，默认为 True。
        timeenc: 时间特征的编码方式，embed默认是timeF，传过来timeenc = 1。
        freq: 时间序列的频率，默认为 'h'（小时）。
        __init__ 方法用于初始化数据集对象，设置各种参数，并调用 __read_data__ 方法来读取数据
        '''
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0] # 命令行参数中的seq_len default 336
            self.label_len = size[1] # default 48
            self.pred_len = size[2] # 预测窗口大小96/192/336/720
        # init
        assert flag in ['train', 'test', 'val'] # if not , raise AssertionError
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag] # 0 1 2

        # args from data_factory.py
        self.features = features # M 多变量预测多变量 预测任务类型
        self.target = target # ‘OT’
        self.scale = scale # scale: 是否对数据进行标准化
        self.timeenc = timeenc # 0 or 1
        self.freq = freq # freq for time features encoding

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        '''
        1. 读取原始数据集，并按照一定规则进行划分成训练集、验证集和测试集
        2. 对数据进行特征工程处理，包括选择特定的特征列、进行标准化处理等
        3. 如果需要，对时间特征进行编码处理，例如提取月份、日期、星期几和小时等信息
        :return:
        '''
        self.scaler = StandardScaler() # 标准化对象
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)) # 读数据集 DataFrame对象实例

        '''
        统一read_csv出的DataFrame中的结构如下：
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)

        '''划分训练集/验证集/测试集划分，训练集和测试集固定7：2，验证集拿剩下的1'''
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len] # [0, 4975, 5735] 训练集/验证集/测试集 所有左边界，num_train - self.seq_len是为了保证验证集至少有1个seq_len输入序列
        border2s = [num_train, num_train + num_vali, len(df_raw)] # [5311, 6071, 7588] 训练集/验证集/测试集 所有右边界 3者区间是有些重叠的
        border1 = border1s[self.set_type] # 0 4975左边界
        border2 = border2s[self.set_type] #5311 6071右边界

        if self.features == 'M' or self.features == 'MS': # 多变量预测多变量/多变量预测单变量
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data] # df_data除去date列
        elif self.features == 'S': # 单变量预测单变量
            df_data = df_raw[[self.target]] # df_data只保留OT列

        # 特征工程
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # 5311 * 8
            self.scaler.fit(train_data.values) # 计算训练数据每个特征的均值和标准差
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values) # 用基于训练集算出来的均值和标准差去标准化整个数据集, 标准化后的整个数据集
        else: # 如果没有特征工程，数据集就不经过标准化
            data = df_data.values

        # 对时间特征编码处理
        df_stamp = df_raw[['date']][border1:border2] # 取训练集/验证集/测试集的date列
        df_stamp['date'] = pd.to_datetime(df_stamp.date) # DataFrame中的日期列转换为pandas的日期时间格式
        if self.timeenc == 0: # fixed, learned 时间编码方式
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1) # 对date列执行函数，对date列的每一行执行row.month操作（获取其月份），存储在名为‘month’的新列中，通过 df_stamp['month'] 进行访问
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1: # timeF时间编码方式
            # fixme:这步根据不同数据集的不同的时间频率freq提取出了此数据集时间在多个时间维度上的时序特征 大小：4 * 5331
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq) # freq default is 'h'
            data_stamp = data_stamp.transpose(1, 0) # 行列转化 4 * 5331 -> 5331 * 4,代表了每一条时序数据在4个时间特征上的值 [...[-0.5,-0.33333333,-0.46666667,-0.49726027]...]

        self.data_x = data[border1:border2] # data：标准化后的数据集 ndarray 5311 * 8
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index): # 重写类的索引操作
        '''
        根据索引获取数据集中的样本
        :param index:
        :return:
        '''
        s_begin = index # 0
        s_end = s_begin + self.seq_len # 336
        r_begin = s_end - self.label_len # 336 - 48 = 288
        r_end = r_begin + self.label_len + self.pred_len # 288 + 48 + 96 = 432

        seq_x = self.data_x[s_begin:s_end] # 索引数据 [0,336]
        seq_y = self.data_y[r_begin:r_end] # 标签数据 [288, 432]
        seq_x_mark = self.data_stamp[s_begin:s_end] # 索引序列时间戳 [0,336]
        seq_y_mark = self.data_stamp[r_begin:r_end] # 标签序列时间戳 [288, 432]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 # train数据集大小 = 5311 - 336 - 96 + 1 = 4880

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    '''
    专用于数据预测的数据集
    '''
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
