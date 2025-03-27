import cv2
import numpy as np
import face_recognition

imgRef = face_recognition.load_image_file("ImagesBasic/ref1.jpg")
imgRef = cv2.cvtColor(imgRef, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("ImagesBasic/test2.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgRef)[0]
encodeRef = face_recognition.face_encodings(imgRef)[0]
cv2.rectangle(imgRef,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeRef],encodeTest)
faceDis = face_recognition.face_distance([encodeRef],encodeTest)
print(results, faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Raoul', imgRef)
cv2.imshow('Test', imgTest)
cv2.waitKey(0)