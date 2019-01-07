import os
import torch
from DORNnet import DORN
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from load_data import getNYUDataset, get_depth_sid


model_path = './run/model_best.pth.tar'
output_dir = './test'


def model_test():
    # 数据
    train_loader, val_loader, test_loader = getNYUDataset()
    # 模型
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    print('模型加载成功')
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()

    for imgs, dpts in train_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            dpts = dpts.cuda()

        dpts_pred, _ = model(imgs)
        dpt_pred_log = dpts_pred[0][0].data.cpu()

        # 预测后的log空间
        print(dpt_pred_log)
        plt.imshow(dpt_pred_log)
        #plt.show()
        plt.imsave(output_dir + '/dpt_pred_log.png', dpt_pred_log)

        # 预测后的实际数值
        dpt_pred = get_depth_sid(dpt_pred_log)
        plt.imsave(output_dir + '/dpt_pred.png', dpt_pred)
        print(dpt_pred)

        # 原始RGB图像
        img = imgs[0].data.cpu().permute(1, 2, 0)
        plt.imsave(output_dir + '/img.png', img)
        #plt.imshow(img)
        #plt.show()
        # print(imgs.size())

        # 原始深度图log空间
        dpt = dpts[0][0].data.cpu()
        plt.imsave(output_dir + '/dpt_log.png', dpt)

        # 原始深度图实际数值
        dpt_sid = get_depth_sid(dpt)
        plt.imsave(output_dir + '/dpt.png', dpt_sid)

        break

if __name__ == '__main__':
    model_test()


