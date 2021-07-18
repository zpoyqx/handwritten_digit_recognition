"""
项目说明：验证手写0-9共10个数字
项目配置：
"""
import torch


train_batch_size = 128
test_batch_size = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")