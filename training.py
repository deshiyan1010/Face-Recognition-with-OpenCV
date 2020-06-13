import cv2
import numpy as np 

def train(face,label):


    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face,np.array(label))

    face_recognizer.save('ymls/trained_faces.yml')

    return face_recognizer

if __name__=="__main__":

    face = np.load("face.npy",allow_pickle=True)
    label = np.load("label.npy",allow_pickle=True)

    train(face,label)