from data_provider.data_factory import data_provider
from experiment.exp_basic import Exp_Basic
from models import DLinear, ZLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
warnings.filterwarnings('ignore') # 忽略所有警告信息
import matplotlib.pyplot as plt
import numpy as np


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args) # args即settings:str，传给Exp_Basic

    def _build_model(self):
        model_dict = { # 键是模型名称，值是模型类
            'DLinear': DLinear,
            'ZLinear': ZLinear,
        }
        # 根据args指定model，调用对应model的构造函数Model()来创建模型实例。通过 .float() 方法将模型转换为浮点类型
        model = model_dict[self.args.model].Model(self.args).float() # Model构造函数在：models/DLinear.py
        # .float() 将张量或模型参数的数据类型转换为 32 位浮点类型（FP32）

        if self.args.use_multi_gpu and self.args.use_gpu: # 多GPU时使用
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        '''
        获取train/val/test的数据
        :param flag: train/val和test
        :return:
        '''
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self): # 选择优化器
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate) #  Adam 优化器
        return model_optim

    def _select_criterion(self): # 选择损失函数
        criterion = nn.MSELoss() # 均方误差损失函数（MSELoss）
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        '''
         在验证集上进行验证，并返回验证集上的损失
        '''
        total_loss = []
        self.model.eval() # 将模型设置为评估模式
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DLinear' or 'ZLinear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DLinear' or 'ZLinear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        '''
        训练模型的方法，接收一个 setting 参数作为模型参数存储路径的一部分，并返回训练好的模型
        '''
        # 获取训练/验证/测试 数据集data_loader
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 创建模型参数存储路径dir ./checkpoints
        path = os.path.join(self.args.checkpoints, setting) # 存模型参数的path默认为./checkpoints/ + 命令行参数字符串
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader) # 在每个训练周期epoch中，训练模型所需的总步数，即每个epoch需要执行多少个训练批次
        # 监控模型在验证集上的损失，如果验证集损失在patience（default 3）次后不再改善，则模型早停（提前结束model train）
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) # 存模型的逻辑在tool.EarlyStopping类function中

        model_optim = self._select_optimizer() # 选择优化器
        criterion = self._select_criterion() # 选择损失函数

        if self.args.use_amp: # GPU 加速，无GPU略过
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs): # 命令行参数指定训练轮数，默认是10轮
            iter_count = 0 # 每次epoch（每次训练迭代）有多少个batch
            train_loss = []

            self.model.train() # 将模型设置为训练模式，每个训练迭代（batch）开始时调用，以准备计算新的梯度
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader): # k:v
                iter_count += 1 # TODO 继续看懂代码
                model_optim.zero_grad() # 将优化器管理的模型参数的梯度归零
                batch_x = batch_x.float().to(self.device) # 输入数据 batch_x 转换为浮点类型，并将其移动到指定的设备上（CPU）

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input 解码器输入数据
                # 列数到倒数第self.args.pred_len列的全0张量
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # 将 batch_y 中的前 self.args.label_len 个时间步的数据拼接到这个全零张量的后面
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp: # GPU
                    with torch.cuda.amp.autocast():
                        if 'DLinear' or 'ZLinear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention: # whether to output attention in ecoder 是否在编码器中输出注意力信息
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else: # 无GPU，用CPU训练
                    if 'DLinear' or 'ZLinear' in self.args.model:
                            outputs = self.model(batch_x) # 使用模型进行前向传播，获取预测结果
                    else: # 非DLinear，其他模型时
                        if self.args.output_attention: # transformer 特有 whether to output attention in ecoder
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0 # 多变量预测单变量
                    outputs = outputs[:, -self.args.pred_len:, f_dim:] # 预测值-截取为预测长度
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) # 真实值-截取为预测长度
                    loss = criterion(outputs, batch_y) # 计算真实值和预测值之间的损失
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0: # 每个训练迭代epoch中每隔 100 个迭代打印一次损失值和训练速度信息
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 根据当前迭代速度和剩余的迭代次数，估算剩余的训练时间
                    speed = (time.time() - time_now) / iter_count # 此次迭代平均速度
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; estimate training left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward() # 计算损失函数对模型参数的梯度，并将其存储在每个参数的 .grad 属性中。
                    model_optim.step() # 根据参数的梯度更新模型参数

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)) # 完成1次epoch的时间
            train_loss = np.average(train_loss) # 对每轮训练批次的模型损失取平均
            vali_loss = self.vali(vali_data, vali_loader, criterion) # 验证集损失
            test_loss = self.vali(test_data, test_loader, criterion) # 测试集损失

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 调用了 EarlyStopping 类的 __call__ 方法，在验证集上进行早停策略的判断。，其中包含save_checkpoint方法，存模型.pth的逻辑
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # 调整优化器学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 训练结束后，从保存的最佳模型参数文件中加载模型的参数
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path)) # 恢复训练过程中保存的最佳模型状态
        # 这样做的目的是在训练过程中使用最佳模型参数来进行模型评估或者后续的预测任务
        return self.model

    def test(self, setting, test=0):
        '''
        Args:
            setting: 命令行参数存成的字符串，是模型参数存储路径的一部分
            test: test=1 时才load模型参数
        在测试集上进行测试，并返回评估指标，如均方误差（MSE）、平均绝对误差（MAE）、相关系数等
        可以设置 test 参数为 1，以加载训练好的模型进行测试。
        '''
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval() # 将模型设置为评估模式
        with torch.no_grad(): # 上下文管理器，用于在其内部禁用梯度跟踪，因为在inference阶段不需要计算梯度，只需要前向传播计算输出结果
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DLinear' or 'ZLinear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DLinear' or 'ZLinear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0: # 每20组 batch记录一次
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0) # ground truth真实值
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0) # pred 预测值
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]) # reshape函数用于调整数组的形状
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds) # 存储预测数据
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        '''
        预测未来数据
        '''
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DLinear' or 'ZLinear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DLinear' or 'ZLinear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
