import os.path

import numpy as np
import cv2
import h5py
import nibabel
import matplotlib.pyplot as plt

def plot1(img_label,gt_label,pred_label,img_path,number1):

    path1 = os.path.join(img_path,str(number1))
    if not os.path.exists(img_path+'/'+str(number1)):

        os.mkdir(img_path+'/'+str(number1))

    # for i in range(img_label.shape[2]):

        # img6 = img_label[:,:,i]

    img9 = img_label * 255


        # img1 = pred_label[:,:,i]

        # img2 = gt_label[:,:,i]

    img9 = img9.astype(np.uint8)

    img1 = pred_label.astype(np.uint8)

    img2 = gt_label.astype(np.uint8)

    image9 = cv2.cvtColor(img9,cv2.COLOR_GRAY2BGR)

    image1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)

    image2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    image92 = image9

    red = [234,51,35]

    green = [117,251,76]

    purple = [91,14,246]

    lblue = [102,220,222]

    lavenderblush = [234,73,247]

    yellow = [254,255,83]

    paleYellow = [251,20,216]

    deeppurple = [0,0,206]

    image91 = np.where(image1 == 1,np.full_like(image9,red),image9)

    image91 = np.where(image1 == 2,np.full_like(image91,lblue),image91)

    image91 = np.where(image1 == 3,np.full_like(image91,green),image91)

    image91 = np.where(image1 == 4,np.full_like(image91,purple),image91)

    image91 = np.where(image1 == 5,np.full_like(image91,lavenderblush),image91)

    image91 = np.where(image1 == 6,np.full_like(image91,yellow),image91)

    image91 = np.where(image1 == 7,np.full_like(image91,paleYellow),image91)

    image9 = np.where(image1 == 8,np.full_like(image91,deeppurple),image91)

    image9 = cv2.cvtColor(image9,cv2.COLOR_BGR2RGB)

    cv2.imwrite(path1+'/'+str(number1)+'_pred'+'.png',image9)

    image9 = image92

    image92 = np.where(image2 == 1,np.full_like(image9,red),image9)

    image92 = np.where(image2 == 2,np.full_like(image92,lblue),image92)

    image92 = np.where(image2 == 3,np.full_like(image92,green),image92)

    image92 = np.where(image2 == 4,np.full_like(image92,purple),image92)

    image92 = np.where(image2 == 5,np.full_like(image92,lavenderblush),image92)

    image92 = np.where(image2 == 6,np.full_like(image92,yellow),image92)

    image92 = np.where(image2 == 7,np.full_like(image92,paleYellow),image92)

    image92 = np.where(image2 == 8,np.full_like(image92,deeppurple),image92)

    image9 = cv2.cvtColor(image92,cv2.COLOR_BGR2RGB)

    cv2.imwrite(path1+'/'+str(number1)+'_gt'+'.png',image9)




if __name__ == '__main__':

    list_dir = '/home/hutao/VesselSeg-Pytorch-master/lists/lists_Synapse/test.txt'

    lists = open(list_dir).readlines()

    if not os.path.exists('/home/hutao/BRAU-Netplusplus-master/' + 'pred1'):
        os.mkdir('/home/hutao/BRAU-Netplusplus-master/' + '/pred1')
    for list1 in lists:

        list2 = '/home/hutao/predictions/'+list1.split('.')[0]+'_pred.nii.gz'

        lists1 = nibabel.load(list2).get_fdata()

        list3 = '/home/hutao/predictions/'+list1.split('.')[0]+'_gt.nii.gz'

        lists2 = nibabel.load(list3).get_fdata()

        list5 = '/home/hutao/predictions/'+list1.split('.')[0]+'_img.nii.gz'


        if not os.path.exists(('/home/hutao/BRAU-Netplusplus-master/'+'pred1'+''+'/'+list1)):

            os.mkdir('/home/hutao/BRAU-Netplusplus-master/pred1'+'/'+''+''+list1)

        lists3 = nibabel.load(list5).get_fdata()

        print(lists1.shape)
        #lists1.permute(2,0,1)
        for i in range(lists1.shape[2]):



            plot1(lists3[:,:,i],lists2[:,:,i],lists1[:,:,i],'/home/hutao/BRAU-Netplusplus-master/pred1'+'/'+list1,i)

