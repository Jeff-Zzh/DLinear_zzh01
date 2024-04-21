import os
import torch
import numpy as np


class Exp_Basic(object):
    '''
    通过继承 Exp_Basic 类，并实现其中的抽象方法，可以定义具体的实验类，
    并在实验类中完成模型的构建、训练、验证和测试等任务。
    '''
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device() # 用cpu还是gpu
        self.model = self._build_model().to(self.device) # 创建一个模型(DLinear/Transformer model)实例，并将其移动到正确的设备(GPU/CPU)上
        # from torchsummary import summary
        # summary(self.model, input_size=(8, 336, 8))


    def _build_model(self): # 抽象方法，给子类去实现初始化model， 在exp_main.py中有实现
        raise NotImplementedError # 强制子类为Exp_Basic类中的抽象方法提供具体的实现
        return None

    def _acquire_device(self):
        '''
        获取设备 GPU CUDA or CPU，本项目只跑DLinear，不跑Transformer，所以CPU即够用
        :return:
        '''
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else: # 在run_longExp.py中args.use_gpu由于torch检测到cuda不是available，将会被设为False
            device = torch.device('cpu') # PyTorch 设备对象
            print('Use CPU')
        return device

    def _get_data(self): # 抽象方法，用于获取数据
        pass

    def vali(self): # 抽象方法，用于在验证集上进行验证
        pass

    def train(self): # 抽象方法，用于训练模型
        pass

    def test(self): # 抽象方法，用于在测试集上进行测试
        pass
