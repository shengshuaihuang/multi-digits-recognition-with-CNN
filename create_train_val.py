import cv2
import numpy as np 
import os
import types
import pickle
import h5py

index = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}

def img2array(imgPath):
    data = []
    for filename in os.listdir(imgPath):
        if filename.split('.')[-1].upper() in ("JPG","JPEG","PNG","BMP","GIF"):
            path = imgPath + '/' + filename
            img = cv2.imread(path,0)
            if img.shape[0]!=55:
                add_row = np.zeros((55-img.shape[0],30))
                img = np.row_stack((img,add_row))
            data.append(img)
    return data


def main():
    data_save = np.empty((0,55,30))
    label_save = np.empty((0,1))
    for folder in ['train','val']:
        imgFolderPath = 'TrainDataset/' + folder
        for i in index:
            imgPath = imgFolderPath + '/' + i
            x = img2array(imgPath)
            y = index[i]*np.ones((len(x),1))
            data_save = np.row_stack((data_save,np.array(x)))
            label_save = np.row_stack((label_save,y))

        data_path = './Network/data/' + folder + '_data.h5'
        label_path = './Network/data/' + folder + '_label.h5'

        f_data = h5py.File(data_path,"w")
        f_label = h5py.File(label_path,"w")
        f_data.create_dataset("data",data_save.shape, data=data_save)
        f_label.create_dataset("label",label_save.shape, data=label_save)
        f_data.close()
        f_label.close()

        data_save = np.empty((0,55,30))
        label_save = np.empty((0,1))

if __name__ == '__main__':
    main()