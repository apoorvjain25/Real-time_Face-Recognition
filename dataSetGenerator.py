import cv2
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('C:\Python27\Internship\Face-Recog\Classifiers/face.xml')
i=0
offset=50
name=raw_input('enter your id')
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("C:\Python27\Face-Recognition\dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.waitKey(100)
    if i>19:
        cam.release()
        cv2.destroyAllWindows()
        break
