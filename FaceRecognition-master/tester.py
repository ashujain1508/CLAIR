import cv2
import os
import numpy as np
import faceRecognition as fr
from gtts import gTTS



def audio():
    my_text = "user found"
    language = 'en'
    myobj = gTTS(text=my_text, lang=language, slow=False)
    myobj.save("welcome.mp3")
    os.system("welcome.mp3")


test_img=cv2.imread('TestImages/divyanshu.jpg')

faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)



faces,faceID=fr.labels_for_training_data('trainingImages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')


name={0:"chirag",1:"divyanshu"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>37):
        continue
        
    fr.put_text(test_img,predicted_name,x,y)
    audio()

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dtecetion ",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows





