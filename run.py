import os

seq_len = 336

print(os.getcwd())
# 创建日志目录
if not os.path.exists("logs"): # 会创建在当前工作目录中
    os.mkdir("logs")
if not os.path.exists("logs/LongForecasting"):
    os.mkdir("logs/LongForecasting")

# 定义运行命令的函数
def run_command(seq_len, pred_len, batch_size, learning_rate):
    command = (
        f"python -u run_experiment.py "
        f"--is_training 1 " 
        f"--root_path ./dataset/ "
        f"--data_path exchange_rate.csv "
        f"--model_id Exchange_{seq_len}_{pred_len} "
        f"--model DLinear "
        f"--data custom " # dataset
        f"--features M " # 多变量预测多变量
        f"--seq_len {seq_len} " # 输入窗口sequence大小
        f"--pred_len {pred_len} "# 预测窗口sequence大小
        f"--enc_in 8 " # number of channels
        f"--des Exp " # 实验说明
        f"--itr 1 " # 实验执行次数（迭代次数）
        f"--batch_size {batch_size} " # batch size of train input data
        f"--learning_rate {learning_rate}" # 学习率
    )
    log_file = f"logs/LongForecasting/DLinear_Exchange_{seq_len}_{pred_len}.log"
    command += f" >{log_file}" # windows中重定向也是> 和 >>
    return command

if __name__ == '__main__':
    # 运行命令并输出日志
    os.system(run_command(seq_len, 96, 8, 0.0005))
    # os.system(run_command(seq_len, 192, 8, 0.0005))
    # os.system(run_command(seq_len, 336, 32, 0.0005))
    # os.system(run_command(seq_len, 720, 32, 0.005))
