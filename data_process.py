import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import shutil

def copy_toGray():
    # 使用opencv读取图片时一定不能有中文路径
    input_dir = 'E:\\m_DORN\\DataSet\\shape1\\Output_depth'
    output_dir = 'E:\\m_DORN\\DataSet\\train_data\\depth'
    index = 0
    # 将三通道的深度图处理成单通道放入目标文件夹中
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('png'):
                print('Being process img %s' % filename)
                img_path = path + '\\' + filename
                #print(img_path)
                img = cv2.imread(img_path)
                #print(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #print(gray.shape)
                cv2.imwrite(output_dir + '\\' + filename, gray)
def copy_img():
    input_dir = 'E:\\m_DORN\DataSet\\形状1\\Output_rgb'
    output_dir = 'E:\\m_DORN\DataSet\\train_data\\rgb'
    # 新建输出文件夹
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('png'):
                img_path = path + '/' + filename
                print('Being process img %s' % filename)
                shutil.copy(img_path, output_dir)



'''
# 读取图片
im = cv2.imread(data_path)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(gray.shape)


for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        if gray[i,j] > 0:
            # im[i,j] = (129, 129, 129)
            print(gray[i,j]) # 这里可以处理每个像素点
            break

print(im.shape)
'''

if __name__ == '__main__':
    copy_toGray()
    # copy_img()
