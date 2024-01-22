import argparse
import os
# globel param
# dataset setting
img_width = 256
img_height = 128
img_channel = 3
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 8
class_num = 2

root_path = os.path.dirname(__file__)
train_path = os.path.join(root_path, 'data/train_index.txt')
val_path = os.path.join(root_path, 'data/val_index.txt')
resize_path = os.path.join(root_path, 'resize')
test_path = os.path.join(root_path, 'picture.txt')  # 深度学习输入文件
save_path = os.path.join(root_path, 'result')  # 深度学习检测结果保存路径
pretrained_path = os.path.join(root_path, 'pretrained/unetlstm.pth')  # 预训练模型路径
save_path_resize = os.path.join(root_path, 'result_resize')


# weight
class_weight = [0.02, 1.02]


def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch UNet-ConvLSTM')
    parser.add_argument('--model', type=str, default='UNet-ConvLSTM',
                        help='( UNet-ConvLSTM | SegNet-ConvLSTM | UNet | SegNet | ')
    parser.add_argument('--batch-size', type=int, default=15, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args
