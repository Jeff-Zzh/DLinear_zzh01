from data_provider.data_loader import Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom, # 自定义数据集 exchange_rate
}


def data_provider(args, flag):
    '''
    真实提供数据的类
    :param args: run_experiment.py中的setting:str(命令行参数)
    :param flag: train/val/test数据集
    :return:
    '''
    Data = data_dict[args.data] # Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1 # embed默认是timeF，时间特征的编码方式


    if flag == 'test':
        shuffle_flag = False # shuffle：是否打乱数据集
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data( # Dataset_Custom
        root_path=args.root_path, # dataset根目录路径
        data_path=args.data_path, # dataset根目录下的数据集相对路径
        flag=flag, # train/val/test/flag
        size=[args.seq_len, args.label_len, args.pred_len], # 输入序列长度，起始标记长度，预测序列长度 336 48 96
        features=args.features, # 多变量预测多变量 预测任务类型
        target=args.target, # 'OT'
        timeenc=timeenc, # 0 or 1
        freq=freq # freq for time features encoding
    )
    print(flag+' set', len(data_set)) # 打印 train 4880     val 665     test 1422
    data_loader = DataLoader(
        data_set, # 要加载的数据集对
        batch_size=batch_size, # 默认 32 每个批次中包含的样本数目为32， 我们传的batch_size为8
        shuffle=shuffle_flag, # 是否在每个 epoch 之前对数据进行洗牌。洗牌可以帮助模型更好地学习数据的分布，提高训练效果。通常在训练过程中会进行洗牌，但在测试过程中则不需要
        num_workers=args.num_workers, # 数据加载器（data loader）中的工作进程数量,用于加载数据
        drop_last=drop_last) # 是否丢弃最后一个批次中不足 batch_size 的样本，通常在训练过程中会设置为 True，而在测试过程中则设置为 False
    return data_set, data_loader
