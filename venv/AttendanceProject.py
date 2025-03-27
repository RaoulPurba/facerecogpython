import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
import pickle  # For saving/loading encodings
from datetime import datetime

path = "ImagesAttendance"
encodings_file = "face_encodings.pkl"  # File to store encodings


# Function to encode images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Function to mark attendance
def markAttendance(name):
    with open("venv/Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtString}")
            print(f"Attendance marked for {name}")


# Load or generate encodings
if os.path.exists(encodings_file):
    with open(encodings_file, "rb") as f:
        encodeListKnown, classNames = pickle.load(f)
    print("Encodings Loaded Successfully")
else:
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f"{path}/{cl}")
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    encodeListKnown = findEncodings(images)

    # Save encodings to file
    with open(encodings_file, "wb") as f:
        pickle.dump((encodeListKnown, classNames), f)
    print("Encodings Saved Successfully")

df = pd.DataFrame(encodeListKnown)
df.insert(0, "Name", classNames)

df.to_csv("encodings.csv", index=False)

print("Encodings saved to encodings.csv")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

            # Draw bounding box
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                img,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Mark attendance
            markAttendance(name)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
