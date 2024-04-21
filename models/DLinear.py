import torch
import torch.nn as nn
# from models import DLinear 试试用自己的DLinear


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    计算时间序列的移动平均，以突出时间序列的趋势性
    """
    def __init__(self, kernel_size, stride):
        '''
        kernel_size:移动平均的窗口大小
        stride:步长
        '''
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) # 计算一维的平均池化

    def forward(self, x):
        """
        前向传播方法接受一个输入张量 x，表示输入的时间序列
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # 将填充后的时间序列传入 nn.AvgPool1d 实例中进行平均池化计算
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    时间序列数据分解模块
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1) # 计算移动平均

    def forward(self, x):
        '''
        前向传播方法
        接受一个输入张量 x，表示输入的时间序列
        '''
        # 首先调用 moving_avg 实例对输入序列进行移动平均计算，得到移动平均值 moving_mean。
        moving_mean = self.moving_avg(x)
        # 计算输入序列 x 与移动平均值之间的残差 res，即原始序列减去移动平均值
        res = x - moving_mean
        return res, moving_mean # 返回元组

class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        '''
        configs形参接收的实参为：self.args，即run_longExp.py中的命令行参数类型argparse.Namespace
        '''
        super(Model, self).__init__()
        self.seq_len = configs.seq_len # 会视窗口大小 default 336
        self.pred_len = configs.pred_len # 预测窗口大小 96/192/336/720

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Decoder = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
