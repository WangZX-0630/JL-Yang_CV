import multiprocessing
import os
import numpy as np
import cv2
from PIL import Image

def convert_file(filename):
    if filename[-3:] == "bmp": #取最后三个字符,读入的是彩色分割图，filename=..rgb-0.bmp
        filepath = "/f2020/shengyifan/data/postdam/test/RGB/" + filename
        savepath = "/f2020/shengyifan/data/postdam/test/gtblack/" + filename[:-4]+"_label.bmp"
        print(filename)
        mask = cv2.imread(filepath, 1) #加载彩色图片，不包括alpha
        (b, g, r) = cv2.split(mask)
        mask = cv2.merge([r, g, b]) #mask维度512*512*3
        label_map = np.zeros([mask.shape[0], mask.shape[1]])
    # surfaces(RGB: 255, 255, 255) 不透水面
    # Building(RGB: 0, 0, 255) 建筑
    # Low vegetation(RGB: 0, 255, 255) 草
    # Tree(RGB: 0, 255, 0) 树
    # Car(RGB: 255, 255, 0) 车
    # Clutter / background(RGB: 255, 0, 0) 背景
        pic = mask #维度512*512*3
        for i in range(pic.shape[0]):
            for j in range(pic.shape[1]):
                if pic[i][j][0] == 255 and pic[i][j][1] == 255 and pic[i][j][2] == 255:
                    label_map[i][j] = 0
                elif pic[i][j][0] == 0 and pic[i][j][1] == 0 and pic[i][j][2] == 255:
                    label_map[i][j] = 1
                elif pic[i][j][0] == 0 and pic[i][j][1] == 255 and pic[i][j][2] == 255:
                    label_map[i][j] = 2
                elif pic[i][j][0] == 0 and pic[i][j][1] == 255 and pic[i][j][2] == 0:
                    label_map[i][j] = 3
                elif pic[i][j][0] == 255 and pic[i][j][1] == 255 and pic[i][j][2] == 0:
                    label_map[i][j] = 4
                elif pic[i][j][0] == 255 and pic[i][j][1] == 0 and pic[i][j][2] == 0:
                    label_map[i][j] = 5
        # label_map = label_map.transpose(2, 0, 1) #转变为6*512*512
        im = Image.fromarray(label_map)
        im=im.convert('L')
        #     # im.show()
        im.save(savepath)

        #   np.save(filepath.split('.')[0], label_map)
        # np.save("/server_space/jiangyl/zhu_useful/Potsdam_512_full/test/gt/"+filename.split('.')[0],label_map)

# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     files = os.listdir("/server_space/jiangyl/zhu_useful/Potsdam_512_full/test/RGB/") #返回指定的文件夹包含的文件或文件夹的名字的列表
#     p = multiprocessing.Pool(10)
#     p.map(convert_file, files)
picpath="/f2020/shengyifan/data/postdam/test/RGB"
piclist=os.listdir(picpath)
for pic in piclist:
    convert_file(pic)
