import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg') # 切换绘图后端


def adjust_learning_rate(optimizer, epoch, args): # 调整学习率
    '''
    根据指定的调整策略来更新优化器中的学习率
    每一次epoch，学习率都不同，每次lr的adjust都和epoch次数有关
    optimizer：优化器对象，用于更新学习率。
    epoch：当前的训练轮次。
    args：命令行参数对象或配置对象，包含了调整学习率的相关参数。
    '''
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1': # defalut 指数衰减策略，每轮学习率按指数衰减的方式变化
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2': # 梯衰减策略，每经过一定的轮次后，将学习率按指定的比例进行缩小
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8 # echo 次数越大， learning_rate越小（每次把learning_rate除2）
        }
    # 其他学习率衰减策略
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    '''
    实现了早停（EarlyStopping）功能，它能够监控模型的验证集损失，
    并在验证集损失不再改善时停止训练，以防止模型过拟合。
    '''
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience # 验证集损失不再改善时，需要等待多少个 epoch 后停止训练
        self.verbose = verbose # 表示是否输出详细信息
        self.counter = 0 # 用于计数连续 epoch 中验证集损失不再改善的次数
        self.delta = delta  # 表示验证集损失相比上次验证集损失的最小改善量，即当验证集损失相比上次验证集损失减少的量小于 delta 时，将计数器增加，default delta=0
        self.best_score = None # 记录最好的验证集损失
        self.early_stop = False # 是否触发了早停
        self.val_loss_min = np.Inf # 记录最小的验证集损失

    def __call__(self, val_loss, model, path): # 使EarlyStopping实例对象可以像调用函数一样使用
        '''
        在每个 epoch 结束时调用，用于更新早停的状态
        如果验证集损失不再改善，则将计数器增加；
        如果计数器达到了 patience 设定的值，则触发早停EarlyStopping。
        如果验证集损失改善了，则保存模型，并更新最小的验证集损失。
        这样可以保证最后在checkpoint.pth中保存的模型权重超参数，是验证集损失最小的模型超参数
        :param val_loss:验证集模型损失
        :param model:要保存的模型
        :param path:保存路径
        :return:
        '''
        score = -val_loss
        if self.best_score is None: # 第一次
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta: # 连续patience次，验证集模型损失都小于这个数值后，模型就没有必要再训练了
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''
        保存模型检查点
        :param val_loss:验证集损失
        :param model:要保存的模型
        :param path:保存路径
        :return:
        '''
        if self.verbose: # 展示详细信息，验证集损失减小了
            print(f'Validation Set loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). \
             Saving model checkpoint.pth...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth') # 保存模型状态字典
        self.val_loss_min = val_loss # 更新model最小验证集损失


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    允许使用点（.）符号访问字典中的属性
    通过此类，我们可以像访问对象的属性一样访问 dotdict 对象的字典属性
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    '''
    用于标准化数据
    用于将数据转换成均值为0，标准差为1的标准正态分布
    '''
    def __init__(self, mean, std):
        self.mean = mean  # 特征的均值
        self.std = std # 特征的标准差

    def transform(self, data):
        '''
        标准化数据的方法
        :param data:接受一个数据数组
        :return:
        '''
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        '''
        反标准化，将标准化后的数据还原成原始数据
        :param data:标准化后的数据数组
        :return:
        '''
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    '''
    Results visualization
    :param true:真实的数据序列
    :param preds: 预测的数据序列
    :param name: 图片保存名称, 这里指定了一个默认路径，但是我们一般会传入要真实存放的路径
    :return:
    '''
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2) # 真实数据
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2) # 预测数据
    plt.legend() # 添加图例标签
    plt.savefig(name, bbox_inches='tight') # 将图片保存为PDF格式

def test_params_flop(model,x_shape):
    '''
    If you want to thest former's flop, you need to give default value to inputs in model.forward(),
    the following code can only pass one argument to forward()
    测试模型的参数量和计算复杂度（FLOPs）

    :param model:模型对象，应该是一个 PyTorch 的 nn.Module 类的子类
    :param x_shape:输入张量的形状，通常是一个元组，表示输入张量的尺寸
    :return:
    '''
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        # 训练模型参数有几百万个
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0): # 有GPU才行
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        # 将计算得到的 FLOPs 和参数数量打印出来。
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))