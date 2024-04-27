# 纯线性神经网络-跑实验模块
import argparse
import os
import torch
from experiment.exp_main import Exp_Main
import random
import numpy as np

# 设置随机数种子
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='DLinear&ZLinear for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='DLinear',
                    help='model name, options: [Autoformer, Informer, Transformer, DLinear, ZLinear]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints') # 存模型权重超参数的地方

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length') # 输入序列长度
parser.add_argument('--label_len', type=int, default=48, help='start token length') # 起始标记长度
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length') # 预测序列的长度


# DLinear/ZLinear args
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times') # 实验次数
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs') # 指定训练轮数
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience') # 验证集损失不再改善时，需要等待多少个 epoch 后停止训练
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description') # 实验描述
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate') # default的优化器学习率调整策略
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

args = parser.parse_args()

# print('Args in experiment:')
# print(args)

Exp = Exp_Main # 类赋值，真正执行实验的类， 后面实例化它去执行实验

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support() # for windows os to run multiprocess use spawn()
    if args.is_training:  # 训练training
        for ii in range(args.itr):  # 迭代几次，命令行给定，默认2   0，1
            # setting:str记录命令行参数, setting将作为模型超checkpoint.pth参数存储路径的一部分
            setting: str = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
                args.model_id,
                args.model, # 模型名 DLinear， ZLinear
                args.data, # dataset type ，custom指自定义数据集
                args.features, # 预测任务类型，多预测多M，多预测单MS，单预测单S
                args.seq_len,# 输入序列长度
                args.label_len, # 起始标记长度
                args.pred_len,# 预测序列的长度
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training setting: {}>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing setting: {}<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:  # 若指定该参数，则predict unseen future data
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

    else:  # 推理 inference
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(args.model_id,
                                                              args.model,
                                                              args.data,
                                                              args.features,
                                                              args.seq_len,
                                                              args.label_len,
                                                              args.pred_len,
                                                              args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
