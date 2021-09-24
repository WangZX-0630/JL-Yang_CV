import cv2,os,random

def cd_function():
    path,dirs,files= next(os.walk("./submits/label"))
    for i in range(1,len(files)+1):
        img1 = cv2.imread("submits/label/ADE_compare_%d.png"%i,-1)
        img2 = cv2.imread("submits/img/ADE_compare_%d.png"%i,-1)
        img3 = cv2.imread("submits/img/ADE_compare_%d.png"%i,3)
        print(i)
        for x in range(img1.shape[0]):
            for y in range(img1.shape[1]):
                if ((img1[x,y]==img2[x,y])and(img1[x,y]==255)):
                    img3[x,y,0]=255
                    img3[x,y,1]=255
                    img3[x,y,2]=255
                elif ((img1[x,y]==img2[x,y])and(img1[x,y]==0)):
                    img3[x,y,0]=0
                    img3[x,y,1]=0
                    img3[x,y,2]=0
                elif ((img1[x,y]!=img2[x,y])and(img1[x,y]==255)):
                    img3[x,y,0]=0
                    img3[x,y,1]=0
                    img3[x,y,2]=255
                elif ((img1[x,y]!=img2[x,y])and(img2[x,y]==255)):
                    img3[x,y,0]=0
                    img3[x,y,1]=255
                    img3[x,y,2]=0
        cv2.imwrite("submits/eval/ADE_eval_%d.png"%i, img3)