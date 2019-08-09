import cv2
import numpy as np
#import matplotlib.pyplot as plt
image=cv2.imread('test_image.jpg')
copy=np.copy(image)
gray=cv2.cvtColor(copy,cv2.COLOR_RGB2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)
canny=cv2.Canny(blur,50,150)
#plt.imshow(canny)
#plt.show()
y=canny.shape[0]
interests=np.array([[[200,y],[1100,y],[550,250]]])
#print(interests.shape)
mask=np.zeros_like(canny)
cv2.fillPoly(mask,interests,255)
masked=cv2.bitwise_and(canny,mask)
lines=cv2.HoughLinesP(masked,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
line_image=np.zeros_like(image)
if lines is not None:
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
final_image=cv2.addWeighted(image,0.8,line_image,1,1)
cv2.imshow("result",final_image)
cv2.waitKey(0)
