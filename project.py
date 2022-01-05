import cv2
import face_recognition
import numpy as np
from datetime import datetime
import os

path="Image"
images = []
classNames=[]
mylist=os.listdir(path)
print(mylist)

for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodingList=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return  encodingList
def markAttandance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListknown=findEncodings(images)
print("Encoding completed")
wCam ,hCam=240,180
url="http://192.168.43.1:8080/video"
cap=cv2.VideoCapture(url)
cap.set(3,wCam)
cap.set(4,hCam)
# cap=cv2.VideoCapture(0)
# cap.set(2,10)
# cap.set(3,20)

while True:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgs)
    encodeCurFrame=face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListknown,encodeFace)
        print(faceDis)

        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1,x1,y2,x2=faceLoc
            y1, x1, y2, x2=y1*4,x1*4,y2*4,x2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            markAttandance(name)

    if img is not None:
        cv2.imshow("frame", img)
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()



    # cv2.imshow('webcam',img)
    # cv2.waitKey(1)





