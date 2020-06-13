import cv2
import numpy as np 
from boundingBox import detect

def label_name(label,name_lst):

    return name_lst[label]

def predict(img,name_lst,face_recognizer):

    #boundedImg,x,y = detect(img)

    label,conf = face_recognizer.predict(np.array(img,dtype=np.uint16))

    name = label_name(label,name_lst)
    #boundedImg = cv2.putText(boundedImg,str(name),(x,y),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    
    return name, conf