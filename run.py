import os

print(os.getcwd())  # D:\PythonProject\DLinear_zzh01
# 创建日志目录
if not os.path.exists("logs"):  # 会创建在当前工作目录中
    os.mkdir("logs")
if not os.path.exists("logs/LongForecasting"):
    os.mkdir("logs/LongForecasting")


# 定义运行命令的函数
def run_command(start_script, data_set, data_set_name, seq_len, pred_len, model, batch_size, learning_rate):
    command = (
        f"python -u {start_script} "
        f"--is_training 1 "
        f"--root_path ./dataset/ "
        f"--data_path {data_set} "
        f"--model_id {data_set_name}_{seq_len}_{pred_len} "  # 记录在哪个数据集上跑且输入序列长度和预测序列长度
        f"--model {model} "
        f"--data custom "  # 自定义dataset
        f"--features M "  # 多变量预测多变量
        f"--seq_len {seq_len} "  # 输入窗口sequence大小
        f"--pred_len {pred_len} "  # 预测窗口sequence大小
        f"--enc_in 8 "  # number of channels
        f"--des Exp "  # 实验说明
        f"--itr 1 "  # 实验执行次数（迭代次数）
        f"--batch_size {batch_size} "  # batch size of train input data
        f"--learning_rate {learning_rate}"  # 学习率
    )
    log_file = f"logs/LongForecasting/{model}_{data_set_name}_{seq_len}_{pred_len}.log"
    command += f" > {log_file}"  # windows中重定向也是> 和 >>
    return command


if __name__ == '__main__':
    # 运行命令并输出日志
    os.system(run_command(start_script='run_experiment.py', data_set='exchange_rate.csv', data_set_name='Exchange',
                          seq_len=336, pred_len=96,
                          model='DLinear', batch_size=8,
                          learning_rate=0.0005))
    # os.system(run_command(seq_len, 192, 8, 0.0005))
    # os.system(run_command(seq_len, 336, 32, 0.0005))
    # os.system(run_command(seq_len, 720, 32, 0.005))
    os.system(run_command(start_script='run_experiment.py', data_set='exchange_rate.csv', data_set_name='Exchange',
                          seq_len=336, pred_len=96,
                          model='ZLinear', batch_size=8,
                          learning_rate=0.0005))
