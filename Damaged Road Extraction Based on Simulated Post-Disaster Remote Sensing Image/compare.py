import cv2,os,random

def compare_function():

    path,dirs,files= next(os.walk("./submits/pre_disaster"))

    for i in range(1,len(files)+1):
        img1 = cv2.imread("submits/pre_disaster/ADE_train_%d.png"%i,-1)
        img2 = cv2.imread("submits/post_disaster/ADE_val_%d.png"%i,-1)
        img3 = img2
        print(i)
        for x in range(img1.shape[0]):
            for y in range(img1.shape[1]):
                if ((img1[x,y]==img2[x,y])and(img1[x,y]==255)):
                    img3[x,y]=0
                elif ((img1[x,y]==img2[x,y])and(img1[x,y]==0)):
                    img3[x,y]=0
                elif ((img1[x,y]!=img2[x,y])and(img1[x,y]==255)):
                    img3[x,y]=255
                elif ((img1[x,y]!=img2[x,y])and(img2[x,y]==255)):
                    img3[x,y]=0
        cv2.imwrite("submits/compare/ADE_compare_%d.png"%i, img3)
if __name__ == "__main__":
    compare_function()