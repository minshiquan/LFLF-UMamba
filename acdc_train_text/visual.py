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

    img1 =cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)

    img2 = gt_label.astype(np.uint8)
    
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    image9 = cv2.cvtColor(img9,cv2.COLOR_GRAY2BGR)

    image92 = image9

    red = [255,0,0]

    green = [0,255,0]

    purple = [128,0,255]

    lblue = [30,244,255]

    #lavenderblush = [234,73,247]

    #yellow = [254,255,83]

    #paleYellow = [251,20,216]

    #deeppurple = [0,0,206]

    image91 = np.where(img1 == 1,np.full_like(image9,red),image9)

    image91 = np.where(img1 == 2,np.full_like(image91,lblue),image91)

    image91 = np.where(img1 == 3,np.full_like(image91,green),image91)

    #image91 = np.where(img1 == 4,np.full_like(image91,purple),image91)

    #image91 = np.where(img1 == 5,np.full_like(image91,lavenderblush),image91)

    #image91 = np.where(img1 == 6,np.full_like(image91,yellow),image91)

    #image91 = np.where(img1 == 7,np.full_like(image91,paleYellow),image91)

    #image9 = np.where(img1 == 8,np.full_like(image91,deeppurple),image91)

    image9 = cv2.cvtColor(image91,cv2.COLOR_BGR2RGB)
    

    cv2.imwrite(path1+'/'+str(number1)+'_pred'+'.png',image9)

    image9 = image92

    image92 = np.where(img2 == 1,np.full_like(image9,red),image9)

    image92 = np.where(img2 == 2,np.full_like(image92,lblue),image92)

    image92 = np.where(img2 == 3,np.full_like(image92,green),image92)

    #image92 = np.where(img2 == 4,np.full_like(image92,purple),image92)

    #image92 = np.where(img2 == 5,np.full_like(image92,lavenderblush),image92)

    #image92 = np.where(img2 == 6,np.full_like(image92,yellow),image92)

    #image92 = np.where(img2 == 7,np.full_like(image92,paleYellow),image92)

    #image92 = np.where(img2 == 8,np.full_like(image92,deeppurple),image92)

    image9 = cv2.cvtColor(image92,cv2.COLOR_BGR2RGB)

    cv2.imwrite(path1+'/'+str(number1)+'_gt'+'.png',image9)




if __name__ == '__main__':

    list_dir = '/home/hutao/BRAU-Netplusplus-master/acdc_train_test/lists_acdc/test.txt'

    lists = open(list_dir).readlines()

    if not os.path.exists('/home/hutao/predictions/ACDC/' + 'pred1'):
        os.mkdir('/home/hutao/predictions/ACDC/' + 'pred1')
    for list1 in lists:

        list2 = '/home/hutao/predictions1/ACDC/TU_ACDC224/TU_pretrain_R50-ViT-B_16_skip3_epo250_bs24_224_ACDC/'+list1.split('.')[0]+'_pred.nii.gz'

        lists1 = nibabel.load(list2).get_fdata()

        list3 = '/home/hutao/predictions1/ACDC/TU_ACDC224/TU_pretrain_R50-ViT-B_16_skip3_epo250_bs24_224_ACDC/'+list1.split('.')[0]+'_label.nii.gz'

        lists2 = nibabel.load(list3).get_fdata()

        list5 = '/home/hutao/predictions1/ACDC/TU_ACDC224/TU_pretrain_R50-ViT-B_16_skip3_epo250_bs24_224_ACDC/'+list1.split('.')[0]+'_img.nii.gz'


        if not os.path.exists(('/home/hutao/predictions/ACDC/'+'pred1'+''+'/'+list1)):

            os.mkdir('/home/hutao/predictions/ACDC/'+'pred1'+''+'/'+list1)

        lists3 = nibabel.load(list5).get_fdata()

        for i in range(lists1.shape[2]):

            
            print(lists3.shape)
            plot1(lists3[:,:,i],lists2[:,:,i],lists1[:,:,i],'/home/hutao/predictions/ACDC/pred1'+'/'+list1,i)


