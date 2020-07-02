import math
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,
                                                                                                               nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


def padding_same_kernel(kernel_size):
    return math.ceil((kernel_size - 1) / 2)


def conv_size(Lin, kernel_size, padding=0, stride=1, dilation=1):
    return int((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def deconv_size(Lin, kernel_size, padding=0, stride=1, dilation=1):
    return (Lin - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1


def padding_same_conv(Lin, kernel_size, stride=1, dilation=1):
    padding = (stride * (Lin - 1) - Lin + dilation * (kernel_size - 1) + 1) / 2
    return int(math.ceil(padding))


def padding_same_deconv(Lin, kernel_size, stride=1, dilation=1):
    padding = kernel_size - padding_same_conv(Lin, kernel_size, stride, dilation) - 1
    return padding


def cal_final_Lout(get_Lout, Lin, kernel_sizes, strides):
    Lout = Lin
    for i in range(len(kernel_sizes)):
        Lout = get_Lout(Lout, kernel_sizes[i], stride=strides[i])

    return Lout
