import os
import torch
import torch.nn as nn
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

cmap = plt.cm.jet
# 加载数据集的index
def load_split():
    current_directoty = os.getcwd()
    train_lists_path = current_directoty + '/data/trainIdxs.txt'
    test_lists_path = current_directoty + '/data/testIdxs.txt'

    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    train_lists = []
    test_lists = []

    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()

    val_start_idx = int(len(train_lists) * 0.8)

    val_lists = train_lists[val_start_idx:-1]
    train_lists = train_lists[0:val_start_idx]

    return train_lists, val_lists, test_lists

# ploy策略的学习率更新
def update_ploy_lr(optimizer, initialized_lr, current_step, max_step, power=0.9):
    lr = initialized_lr * ((1 - float(current_step) / max_step) ** (power))
    idx = 0
    for param_group in optimizer.param_groups:
        if idx == 0: # base params
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * 20
        # print(idx, lr, torch.cuda.current_device())
        idx += 1
    return lr

# 定义序数损失函数
class ordLoss(nn.Module):
    def __init__(self):
        super(ordLoss, self).__init__()
        self.loss = 0.0
    def forward(self, ord_labels, target):
        N, C, H, W = ord_labels.size()
        ord_num = C
        if torch.cuda.is_available():
            K = torch.zeros((N, C, H, W), dtype=torch.float32).cuda()
            for i in range(ord_num):
                K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.float32).cuda()
        else:
            K = torch.zeros((N, C, H, W), dtype=torch.float32)
            for i in range(ord_num):
                K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.float32)

        mask_0 = (K <= target).detach() # 分类正确
        mask_1 = (K > target).detach() # 分类错误

        one = torch.ones(ord_labels[mask_1].size())
        if torch.cuda.is_available():
            one = one.cuda()

        self.loss = torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-7, max=1e7))) \
                    + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-7, max=1e7)))

        N = N * H * W
        self.loss /= (-N)
        return self.loss

# 保存检查点
def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


# 保存结果图片
def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C

def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
