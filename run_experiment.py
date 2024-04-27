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


# DLinear args
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Transformer args
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='for transformer model:dimension of model') # d_model 是模型的隐藏层维度，也称为模型的表示空间的维度。在Transformer中，它代表了注意力机制中的查询、键和值的维度，以及全连接层的维度。
parser.add_argument('--n_heads', type=int, default=8, help='for transformer model:num of heads') # n_heads 是注意力头的数量。在自注意力机制中，输入被投影到不同的子空间，每个头负责不同的注意力计算。增加头的数量可以增加模型的表达能力和并行性。
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') # e_layers 是编码器的层数，控制了编码器中重复堆叠的自注意力层和前馈神经网络层的数量。增加层数可以增加模型的表示能力，但也可能增加训练和推理的计算成本。
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') # d_layers 是解码器的层数，控制了解码器中重复堆叠的自注意力层、编码器-解码器注意力层和前馈神经网络层的数量。同样地，增加层数可以增加模型的表示能力，但也可能增加计算成本。
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn') # 前馈神经网络中隐藏层的维度大小，也称为全连接网络的维度。增加这个值可以增加模型的表示能力，但也会增加计算成本。2048 是一个常用的值，但在某些情况下，可能需要根据任务和数据集进行调整。
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average') # 移动平均模块的窗口大小，用于计算时间序列的移动平均值。较大的窗口大小可以平滑时间序列并突出趋势，但可能会丢失一些细节信息，较小的窗口大小则可能更加敏感于噪声。选择合适的窗口大小通常需要根据数据集的特征和任务需求进行调整。
parser.add_argument('--factor', type=int, default=1, help='attn factor') # 控制注意力机制中的缩放因子（scaling factor）影响了注意力权重的计算
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True) # 控制是否在编码器中使用知识蒸馏（distillation）技术。知识蒸馏是一种模型压缩技术，通过让一个大型模型（教师模型）传递其知识给一个小型模型（学生模型），来提高学生模型的性能和泛化能力。在这里，如果设置为 True，则表示使用知识蒸馏，如果设置为 False，则表示不使用知识蒸馏。
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', # 时间特征的编码方式，影响了模型如何处理时间信息
                    help='time features encoding, options:[timeF, fixed, learned]')
'''
timeF 表示时间特征会被编码为固定的正弦和余弦函数，这种编码方式能够捕捉时间的周期性和周期之间的关系，适合于处理周期性时间序列数据。
fixed 表示时间特征会被编码为固定的向量，不考虑时间的周期性，适合于处理非周期性时间序列数据。
learned 表示时间特征会被模型自动学习编码，模型将从数据中学习到时间特征的表示，这种方式更加灵活，但可能需要更多的数据来训练模型。
'''
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder') # 是否在编码器中输出注意力信息
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers') # 指定数据加载器（data loader）中的工作进程数量,多个工作进程并行地加载数据，从而加快数据的加载速度
parser.add_argument('--itr', type=int, default=2, help='experiments times') # 实验次数
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs') # 指定训练轮数
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience') # 验证集损失不再改善时，需要等待多少个 epoch 后停止训练
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description') # 实验描述
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate') # default的优化器学习率调整策略
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# 在这一步use_gpu参数会被设为False，DLinear/ZLinear很小，只用cpu跑
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False # 命令行指定用gpu，并且torch framework觉得跑当前程序的DUT的cuda是能用的，才把该标志位设为True

# 多GPU training/inference情况
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# print('Args in experiment:')
# print(args)

Exp = Exp_Main # 类赋值，真正执行实验的类， 后面实例化它去执行实验

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support() # for windows os to run multiprocess use spawn()
    if args.is_training:  # 训练training
        for ii in range(args.itr):  # 迭代几次，命令行给定，默认2   0，1
            # setting:str记录命令行参数, setting将作为模型超checkpoint.pth参数存储路径的一部分
            setting: str = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model, # 模型名 DLinear， ZLinear
                args.data, # dataset type ，custom指自定义数据集
                args.features, # 预测任务类型，多预测多M，多预测单MS，单预测单S
                args.seq_len,# 输入序列长度
                args.label_len, # 起始标记长度
                args.pred_len,# 预测序列的长度
                args.d_model, # 用于Transformer模型或其变体，用于控制模型的维度 dimension of model 模型维度 默认512
                args.n_heads, # 用于Transformer模型或其变体，用于控制注意力头的数量 num of heads 模型维度 默认8
                args.e_layers, # Transformer 编码器层数
                args.d_layers,# Transformer 解码器层数
                args.d_ff, # 前馈神经网络中隐藏层的维度大小
                args.factor, # 控制注意力机制中的缩放因子（scaling factor）影响了注意力权重的计算
                args.embed, # 时间特征的编码方式，影响了模型如何处理时间信息
                args.distil, # 是否在编码器中使用知识蒸馏（distillation）技术
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
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
