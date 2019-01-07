import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import h5py
from PIL import Image
import numpy as np
from m_utils import load_split

# data_path = './data/nyu_depth_v2_labeled.mat'
data_path = './DataSet'
batch_size = 2
iheight, iwidth = 480, 640 # raw image size
alpha, beta = 0.02, 10.02
K = 68
output_size = (257, 353)

# FireWorkDataSet
class FireWork_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, type='train'):
        example = 0
        if type == 'train':
            # 统计当前文件的数量
            DIR = data_path + '/' + type + '_data/rgb'
            example = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

        imgs = []
        dpts = []
        for i in range(example):
            img_path = data_path + '/' + type + '_data/rgb/' +  str(i) + '.png'
            imgs.append(img_path)
            dpt_path = data_path + '/' + type + '_data/depth/' + str(i) + '.png'
            dpts.append(dpt_path)

        self.imgs = imgs
        self.dpts = dpts

    def __getitem__(self, index):
        img_path = self.imgs[index]
        dpt_path = self.dpts[index]

        img = Image.open(img_path)
        dpt = Image.open(dpt_path)

        img_transform = transforms.Compose([
            transforms.Resize(output_size),
            transforms.ToTensor()
        ])

        img = img_transform(img)
        dpt = img_transform(dpt)
        dpt = scale(dpt)
        dpt = get_depth_log(dpt)
        return img, dpt

    def __len__(self):
        return len(self.imgs)

# 将深度图缩放10倍,深度的范围就是（0-10m)进一步操作（0.02-10.02）
def scale(depth):
    ratio = torch.FloatTensor([10.0])
    offset = torch.FloatTensor([0.02])
    return ratio * depth + offset

# 加载NYU_mat类型数据集
class NYU_Dataset(data.Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists
        self.nyu = h5py.File(self.data_path)
        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']
        self.output_size = (257, 353)

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0) #HWC
        dpt = self.dpts[img_idx].transpose(1, 0)
        img = Image.fromarray(img)
        dpt = Image.fromarray(dpt)
        img_transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])
        dpt_transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])
        img = img_transform(img)
        dpt = dpt_transform(dpt)
        # 将深度图变化到对数空间
        dpt = get_depth_log(dpt)
        return img, dpt

    def __len__(self):
        return len(self.lists)


#从(0,K)->(alpha, beta)
def get_depth_log(depth):
    alpha_ = torch.FloatTensor([alpha])
    beta_ = torch.FloatTensor([beta])
    K_ = torch.FloatTensor([K])
    t = K_ * torch.log(depth / alpha_) / torch.log(beta_ / alpha_)
    # t = t.int()
    return t

# 从(alpha,beta)->(0,K)
def get_depth_sid(depth_labels):
    depth_labels = depth_labels.data.cpu()
    alpha_ = torch.FloatTensor([alpha])
    beta_ = torch.FloatTensor([beta])
    K_ = torch.FloatTensor([K])
    t = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * depth_labels / K_)
    return t

def getNYUDataset():
    train_lists, val_lists, test_lists = load_split()

    train_set = NYU_Dataset(data_path=data_path, lists=train_lists)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    val_set = NYU_Dataset(data_path=data_path, lists=val_lists)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)

    test_set = NYU_Dataset(data_path=data_path, lists=test_lists)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader


import matplotlib
import matplotlib.pyplot as plt

def load_test():
    # test_set = NYU_Dataset(data_path=data_path, lists=test_lists)
    # test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    train_data = FireWork_Dataset(data_path, type='train')
    train_loader = data.DataLoader(train_data, batch_size=4, shuffle=True)
    for imgs, dpts in train_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            dpts = dpts.cuda()
        img = imgs[0].data.cpu().permute(1, 2, 0)
        plt.imshow(img)
        # plt.show()
        dpt = dpts[0][0].data.cpu()
        print(dpt)

        for i in range(dpt.size(0)):
            for j in range(dpt.size(1)):
                if dpt[i][j] > 0:
                    print(dpt[i][j])

        plt.imshow(dpt)
        plt.show()
        print(imgs.size())
        print(dpts.size())

        #plt.imsave('./data/dpt1.png', dpt)
        #plt.imshow(dpt)
        #plt.show()
        break

if __name__ == '__main__':
    #a = torch.FloatTensor([68]).cuda()
    #print(get_depth_sid(a))
    load_test()
