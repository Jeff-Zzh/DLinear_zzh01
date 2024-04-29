import torch
import torch.nn as nn
# from models import DLinear 试试用自己的DLinear


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    移动平均模块，计算时间序列的移动平均，以突出时间序列的趋势性
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
        # padding on the both ends of time series 对输入的时间序列进行填充
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out += residual  # 添加残差连接
        out = self.relu(out)
        out = self.dropout(out)
        return out

class Model(nn.Module):
    """
    ZLinear：整合移动平均模块和时间序列分解模块，使用线性层 来预测 时间序列的趋势 和 季节性成分
    使用线性层来预测时间序列的趋势和季节性成分
    """
    def __init__(self, configs):
        '''
        configs形参接收的实参为：self.args，即run_longExp.py中的命令行参数类型argparse.Namespace
        '''
        super(Model, self).__init__()
        self.seq_len = configs.seq_len # 回视窗口大小（输入时间序列的窗口大小） default 336
        self.pred_len = configs.pred_len # 预测窗口大小 96/192/336/720

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size) # 时间序列分解的移动平均模块的窗口大小
        self.individual = configs.individual # 命令行参数 DLinear: a linear layer for each variate(channel) individually 标志着是否为每个通道（变量）使用单独的线性层
        self.channels = configs.enc_in # 表示输入数据中的通道数

        # self.conv1 = nn.Conv1d(self.channels, 64, kernel_size=3, padding=1)  # 增加卷积层
        # self.residual_blocks = nn.ModuleList([ResidualBlock(64, 64, kernel_size=3) for _ in range(2)])  # 引入残差块

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
        # Additional layers
        self.dropout = nn.Dropout(0.01) # probability of an element to be zeroed. Default: 0.1
        # self.relu = nn.ReLU()

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x) # 调用时间序列分解模块得到趋势和季节性成分
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        if self.individual: # 为真，为每个通道分别应用线性层；否则，所有通道共享同一线性层
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels): # 对于每个通道，都有一个季节性线性层 Linear_Seasonal 和一个趋势线性层 Linear_Trend，分别用于预测季节性成分和趋势
                # 使用全连接层得到季节性
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                # 使用全连接层得到趋势性
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
                # 两者共享所有权重
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # Additional layers and activations
        # seasonal_output = self.relu(seasonal_output)
        seasonal_output = self.dropout(seasonal_output)
        # trend_output = self.relu(trend_output)
        trend_output = self.dropout(trend_output)

        x = seasonal_output + trend_output # 将季节性输出和趋势输出相加，得到最终的预测结果
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
