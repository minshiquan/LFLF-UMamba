import os.path

import cv2
import numpy  as np
import matplotlib.pyplot as plot1

import pandas as pd

def skin_plot(img_add):
    
    img_add1 = img_add.split('/')[-1]

    img_add2 = img_add.split('B')[0]+'BRAU-Netplusplus-master/debug2/'+img_add.split('/')[-1].split('.')[0]+'_pred.png'

    img_add3 = img_add.split('B')[0]+'BRAU-Netplusplus-master/debug2/'+img_add.split('/')[-1].split('.')[0]+'_gt.png'
    
    img = cv2.imread(img_add)

    img1 = cv2.imread(img_add2)

    img2 = cv2.imread(img_add3)

    print(img_add3)
    # exit()
    img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)

    img1_contours,img1_hierarchy = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    img2_contours,img2_hierarchy = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for img2_cont in img2_contours:

        cv2.drawContours(img,[img2_cont],-1,(255,0,0), 1)

    for img1_cont in img1_contours:

        cv2.drawContours(img,[img1_cont],-1,(0,0,255), 1)
    for img_add5 in img_add.split('/'):
        if 'home' in img_add5:
            img_add3 = ''
            img_add3 += '/'+img_add5 + '/'
        if '.png' not in img_add5 and img_add5 != 'home':
            img_add3 += img_add5 + '/'


    # img3 = img_add.split('/')[0]
    print(img_add3)
    cv2.imwrite(img_add3+'contour_'+img_add1.split('.')[0]+'.png',img)

if __name__ == '__main__':

     df = pd.read_csv('./src/test_train_data.csv')

     test_df1 = df[df.category == 'test']

     test_file1 = list(test_df1.image_id)

     print(len(test_file1))
    
     for file1 in  test_file1:
        # print(file1)
        image_path = os.path.join('/home/hutao/BRAU-Netplusplus-master/debug2',file1.split('.')[0]+'.png')
        skin_plot(image_path)
