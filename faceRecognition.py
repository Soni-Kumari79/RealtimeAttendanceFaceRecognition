import cv2
import numpy as np
import os 
import mysql.connector
import datetime


def insertAttendance(face_id):
	mydb = mysql.connector.connect(
		host="localhost",
		user="root",
		password="sabina",
		database="face_attendance"
	)
	mycursor = mydb.cursor()
	d = datetime.datetime.now()
	cd = d.strftime("%d") + "/" + d.strftime("%m") + "/" + d.strftime("%Y")
	sql = "SELECT * FROM attendance WHERE RegID = %s AND atten_date = %s"
	val = (face_id, cd)
	mycursor.execute(sql, val)
	# print(mycursor.fetchone())
	if mycursor.fetchone() == None:
		sql = "INSERT INTO attendance (RegID, atten_date) VALUES (%s, %s)"
		val = (face_id,cd)
		mycursor.execute(sql, val)
		mydb.commit()
		print(mycursor.rowcount, "record updated.")

def fetchName(face_id):
	mydb = mysql.connector.connect(
		host="localhost",
		user="root",
		password="sabina",
		database="face_attendance"
	)
	mycursor = mydb.cursor()
	sql = "SELECT Name FROM users WHERE RegID = %s"
	val = (face_id, )
	mycursor.execute(sql, val)
	return mycursor.fetchone()[0]


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
# names = ['None', 'Sabina', 'Sabina', 'Mayank'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    # img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            regID = id
            id = fetchName(regID)
            confidence = "  {0}%".format(round(100 - confidence))
            insertAttendance(regID)
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()