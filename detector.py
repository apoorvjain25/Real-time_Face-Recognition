import cv2,os
import numpy as np
from PIL import Image 
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\Python27/trainer.yml')
cascadePath = "C:\Python27\Internship\Face-Recog\Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'C:\Python27\Internship\Face-Recog/dataSet'

cam = cv2.VideoCapture(0)


#font = cv2.initFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 4) #Creates a font	
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale=1
fontcolor= (255, 255, 255)


id=0
#names = ['None', 'Apoorv', 'Paula', 'Ilza', 'Z', 'W'] 

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        #nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
	id, conf = recognizer.predict(gray[y:y+h,x:x+w])

	cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        #if(id==1):
	#	 id='Apoorv'
	#else:
	#    id='unknown'
        #cv2.cv.putText(cv2.fromarray(im),str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255) #Draw the text
        #cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255, 2) #Draw the text
	cv2.putText(im,str(id), (x,y+h),font, 255, 2)
	cv2.putText(im, str(conf), (x,y+h), font, 1, 255, 1)
	cv2.putText(im, str(id), (x-5,y+h+5), font, fontscale, fontcolor)
	cv2.imshow('im',im)
        cv2.waitKey(10)
cv2.destroyAllWindows()

      #  elif(nbr_predicted==1):
       #      nbr_predicted='Apoorv'






