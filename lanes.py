import cv2
import numpy as np
#import matplotlib.pyplot as plt

def coor(image,para):
    m,c=para
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-c)/m)
    x2=int((y2-c)/m)
    return np.array([x1,y1,x2,y2])
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
#final_image=cv2.addWeighted(image,0.8,line_image,1,1)
left_fit=[]
right_fit=[]
for line in lines:
    x1,y1,x2,y2=line.reshape(4)
    para=np.polyfit((x1,x2),(y1,y2),1)
    m=para[0]
    c=para[1]
    if m<0:
        left_fit.append((m,c))
    else:
        right_fit.append((m,c))
left_fit_average=np.average(left_fit,axis=0)
right_fit_average=np.average(right_fit,axis=0)
left_line=coor(image,left_fit_average)
right_line=coor(image,right_fit_average)
new_line=np.array([left_line,right_line])
new_line_image=np.zeros_like(image)
if new_line is not None:
    for line in new_line:
        x1,y1,x2,y2=line.reshape(4)
        cv2.line(new_line_image,(x1,y1),(x2,y2),(255,0,0),10)
new_image=cv2.addWeighted(image,0.8,new_line_image,1,1)
cv2.imshow("result",new_image)
cv2.waitKey(0)
