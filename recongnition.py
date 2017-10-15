#coding=utf-8
import os
import numpy as np
import cv2
import xlrd
from xlwt import *
import pickle
import h5py

from skimage import data,filters,segmentation,measure,morphology,color

from keras.preprocessing import image
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

#something define
src_folder = 'RawDataset/Images'
tar_folder = 'Result'
if not os.path.exists(tar_folder):
	os.mkdir(tar_folder)
w = Workbook(encoding = 'utf-8')
ws = w.add_sheet('result')
ws.write(0, 0, 'Filename')
ws.write(0, 1, 'Ground Truth')
ws.write(0, 2, 'Prediction')

gndtruth = pickle.load(open('./Network/data/gndtruth.pkl','rb'))


#loading the predict model
json_file = open('./Network/model/digit_CNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./Network/model/digit_CNN.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# pre-processing  binary and transform img to array with 1 and 0
def img_to_binary(img):
	blur = cv2.medianBlur(img,5)
	cv2.threshold(blur, 90, 255, 0, blur)
	img_binary=np.array(blur)
	for i in range(img_binary.shape[0]):
		for j in range(img_binary.shape[1]):
			if img_binary[i,j]==0:
				img_binary[i,j]=1
			else:
				img_binary[i,j]=0
	return img_binary


# roughly crop the region of the digit
def get_digits_region(im):
	left=[]
	right=[]
	top = []
	bottom = []
	weight = im.shape[1]/2
	for i in range(im.shape[0]-1):
		if sum(im[i,:])<weight and sum(im[i+1,:])>weight:
			top.append(i)
		elif sum(im[i,:])>weight and sum(im[i+1,:])<weight:
			bottom.append(i)
	try:
		top = [val for val in top if val>150 and val<250]
		t = min(top)
		bottom = [val for val in bottom if val>250 and val<300]
		b = max(bottom)
	except:
		t = 195
		b = 280
	return [0,im.shape[1],t,b]


# by using HoughLines, precisely crop the region of digits and get the edges of the digit.
def get_digits_edges(img):
	img = cv2.GaussianBlur(img,(3,3),0)
	canny = cv2.Canny(img, 50, 150, apertureSize = 3)
	lines = cv2.HoughLines(canny,1,np.pi/180,118) #这里对最后一个参数使用了经验型的值  
	result = img.copy()
	rect = []
	for line in lines[0]:
		rho = line[0] #第一个元素是距离rho  
		theta= line[1] #第二个元素是角度theta  
		if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线  
			#该直线与第一行的交点  
			pt1 = (int(rho/np.cos(theta)),0)
			#该直线与最后一行的焦点  
			pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
			cv2.line( result, pt1, pt2, (255))
		else: #水平直线  
			# 该直线与第一列的交点  
			pt1 = (0,int(rho/np.sin(theta)))
			#该直线与最后一列的交点  
			pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))  
			cv2.line(result, pt1, pt2, (255), 1)
		rect.append(pt1)
		rect.append(pt2)
	try:
		top = max([val[1] for val in rect if val[1]<50])
	except:
		top = 0
		print('top')
	try:
		bottom = min([val[1] for val in rect if val[1]>40])
	except:
		bottom = img.shape[1] -5
		# print('bottom')
	crop_img = img[top:bottom,50:415]
	edges = cv2.Canny(crop_img, 50, 150, apertureSize = 3)
	return edges


def get_all_single_digit(img,filename):
	rects = []
	thresh =filters.threshold_otsu(img) #阈值分割
	bw =morphology.closing(img > thresh, morphology.square(3)) #闭运算
	cleared = bw.copy()  #复制
	segmentation.clear_border(cleared)  #清除与边界相连的目标物
	label_image =measure.label(cleared)  #连通区域标记
	borders = np.logical_xor(bw, cleared) #异或
	label_image[borders] = -1
	# image_label_overlay =color.label2rgb(label_image, image=image) #不同标记用不同颜色显示
	for region in measure.regionprops(label_image): #循环得到每一个连通区域属性集
		#忽略小区域
		if region.area < 60:
			continue
		#绘制外包矩形
		minr, minc, maxr, maxc = region.bbox
		width = maxc - minc
		height = maxr - minr
		if width > 26 or width < 5 or height <20:
			continue
		rects.append([int(minc-2),int(maxc+2),int(minr),int(maxr)])
	rects.sort()
	remove_indices = []
	for i in range(len(rects)-1):
		if (rects[i][3] - rects[i+1][2]) < 5:
			# remove_indices.append(i+1)
			rects[i+1][3] = rects[i+1][2]
			rects[i+1][2] = 5
		if abs(rects[i][0]-rects[i+1][0]) + abs(rects[i][1]-rects[i+1][1]) < 10:
			if rects[i][3] <= rects[i+1][3]:
				remove_indices.append(i)
			else:
				remove_indices.append(i+1)
	remove_indices = sorted(set(remove_indices),key=remove_indices.index)
	rects = [i for j, i in enumerate(rects) if j not in remove_indices]
	return rects

def add_padding(img):
	height,width = img.shape
	gapped = np.empty((55,30))
	gap_h = 30 - width
	gap_v = 55 - height
	left  = int(gap_h/2)
	top  = int(gap_v/2)
	gapped[:,0:left] = 0
	gapped[:,left + img.shape[1]:] = 0
	if gap_v<=0:
		gapped[:,left:left + width] = img[:35,:]
	else:
		gapped[0:top,:] = 0
		gapped[top:top + height,left:left + width] = img
		gapped[top + height:,:] = 0
	return gapped


def predict(digit):
	digit = digit.reshape(1,55,30)
	digit = digit.astype('float32')
	digit /=255
	digit = np.expand_dims(digit, axis=0)
	pre = loaded_model.predict(digit)
	return str(np.where(pre == np.max(pre))[1][0])


def write_result(result, threshold, index, filename,imgsrc):
	putText = ''.join(result[:5]) + '.' + ''.join(result[5:])
	font = cv2.FONT_HERSHEY_SIMPLEX
	ws.write(index+1, 0, filename)
	ws.write(index+1, 1, gndtruth[filename])
	ws.write(index+1, 2, putText)

	if putText[0:threshold] == gndtruth[filename][0:threshold]:
		cv2.putText(imgsrc,putText,(10,50), font, 2,(255,195,25),2)
		save_folder = os.path.join(tar_folder, 'correct')
		if not os.path.exists(save_folder):
			os.mkdir(save_folder)
		save_path = os.path.join(save_folder, filename)
		cv2.imwrite(save_path,imgsrc)
		return 0
	else:
		ws.write(index,3,'error')
		cv2.putText(imgsrc,putText,(10,50), font, 2,(0,0,255),2)
		save_folder = os.path.join(tar_folder, 'error')
		if not os.path.exists(save_folder):
			os.mkdir(save_folder)
		save_path = os.path.join(save_folder, filename)
		cv2.imwrite(save_path,imgsrc)
	cv2.imwrite(save_path, imgsrc)
	return 1


def handle_single_Image(filename,tar,index,threshold):

	if threshold >7 or threshold <=0:
		print("threshold should be 1 to 7!")
	imgp_path = os.path.join(src_folder, filename)
	# open the pic
	imgsrc = cv2.imread(imgp_path)
	img = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2GRAY)
	# pre-processing and cropping
	region_rectangle = get_digits_region(img_to_binary(img))
	croped_img = img[region_rectangle[2]:region_rectangle[3],region_rectangle[0]:region_rectangle[1]]

	# get the edges of digit
	edges = get_digits_edges(croped_img)

	# get all single-digits applied Canny with a list formatted [rectange0, rectange1,....]
	digit_mark = get_all_single_digit(edges,filename)


	multi_digit = []
	for i in range(len(digit_mark)):
		gap = 30-digit_mark[i][1]+digit_mark[i][0]
		digit_raw = edges[digit_mark[i][2]:digit_mark[i][3],digit_mark[i][0]:digit_mark[i][1]]

		if sum(sum(digit_raw)) > 500:
			digit = add_padding(digit_raw)
			multi_digit.append(digit)

	result=[]
	for j, digit in enumerate(multi_digit):
		if j<7:
			result.append(predict(digit))

	return write_result(result, threshold, index, filename, imgsrc)


def main():
	index = 0
	err = 0
	threshold = 7
	for filename in os.listdir(src_folder):
		if filename.split('.')[-1].upper() in ("JPG","JPEG","PNG","BMP","GIF"):
			err = err + handle_single_Image(filename,tar_folder,index, threshold)
			index +=1
	print('Error rate:%s/%s=%s'%(err,index,round(err)/index))

if __name__ == '__main__':
	main()
	w.save('result.xls')