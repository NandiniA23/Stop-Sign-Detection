import numpy as np
import cv2

kernel=np.ones((3,3),np.uint8)
cap=cv2.VideoCapture(0)

while (1):
                ret,img=cap.read()
                
                ori=img.copy()
                hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                mask=cv2.inRange(hsv,(170,0,0),(180,255,255))
                #cv2.imshow("Mas",mask)

                erode=cv2.erode(mask,kernel,iterations=1)
                #cv2.imshow("Erode",erode)
                dialate=cv2.dilate(erode,kernel,iterations=3)
                #cv2.imshow("Dialate",dialate)
                contours,heirarchy=cv2.findContours(dialate,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                max_area_contour=max(contours,key=cv2.contourArea)
                img1=cv2.drawContours(img,[max_area_contour],0,(255,0,0),4)
                #cv2.imshow("Final1",img1)
                hull=cv2.convexHull(max_area_contour)
                solidity=cv2.contourArea(max_area_contour)/cv2.contourArea(hull)
                #print(solidity)
                if solidity>0.9:
                        
                        epsilon=0.005*cv2.arcLength(max_area_contour,True)
                        approx=cv2.approxPolyDP(max_area_contour,epsilon,True)
                        #print(len(approx))
                        final=cv2.drawContours(ori,[approx],0,(0,255,0),4)
                        #cv2.imshow("Final",final)
                        if len(approx)==8: 
                                        x,y,w,h=cv2.boundingRect(approx)
                                        final=cv2.rectangle(ori,(x,y),(x+w,y+h),(255,0,0),3)
                                        cv2.imshow("Final",final)
                else:
                        
                        cv2.imshow("Final",ori)
                k=cv2.waitKey(1)
                if k==27:
                         break

        




                  
