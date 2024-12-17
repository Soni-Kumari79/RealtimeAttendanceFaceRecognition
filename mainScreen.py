import datetime
from os import system, name
import cv2
import os
from PIL import Image
import numpy as np
import mysql.connector

# Defining functions
# Function for creating dataset and training
def faceDataset(face_id):
	cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	cam.set(3, 640) # set video width
	cam.set(4, 480) # set video height
	face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
	
	print("\n [INFO] Initializing face capture. Look the camera and wait ...")
	# Initialize individual sampling face count
	count = 0
	while(True):
		ret, img = cam.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_detector.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
			count += 1
			# Save the captured image into the datasets folder
			cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
			cv2.imshow('image', img)
		k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
		if k == 27:
			break
		elif count >= 30: # Take 30 face sample and stop video
			break
	# Do a bit of cleanup
	print("\n [INFO] Finishing Datasets and cleanup stuff")
	cam.release()
	cv2.destroyAllWindows()
	
def getImagesAndLabels(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
	faceSamples=[]
	ids = []
	detector = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml");
	for imagePath in imagePaths:
		PIL_img = Image.open(imagePath).convert('L') # grayscale
		img_numpy = np.array(PIL_img,'uint8')
		id = int(os.path.split(imagePath)[-1].split(".")[1])
		faces = detector.detectMultiScale(img_numpy)
		for (x,y,w,h) in faces:
			faceSamples.append(img_numpy[y:y+h,x:x+w])
			ids.append(id)
	return faceSamples,ids

def faceTraining():
	path = 'dataset'
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	
	print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
	faces,ids = getImagesAndLabels(path)
	recognizer.train(faces, np.array(ids))
	# Save the model into trainer/trainer.yml
	recognizer.write('trainer/trainer.yml') 
	# Print the numer of faces trained and end program
	print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
	

def insertUserDatabase(face_id,face_name):
	mydb = mysql.connector.connect(
		host="localhost",
		user="root",
		password="sabina",
		database="face_attendance"
	)
	mycursor = mydb.cursor()
	sql = "INSERT INTO users (RegID, Name) VALUES (%s, %s)"
	val = (face_id,face_name)
	mycursor.execute(sql, val)
	mydb.commit()
	print(mycursor.rowcount, "record updated.")
	
def reportToday(cur_date):
	mydb = mysql.connector.connect(
		host="localhost",
		user="root",
		password="sabina",
		database="face_attendance"
	)
	mycursor = mydb.cursor()
	sql = "SELECT RegID FROM attendance WHERE atten_date = %s"
	val = (cur_date, )
	mycursor.execute(sql, val)
	myresult = mycursor.fetchall()
	print("Attendance Report for " + cur_date + ":")
	# print("Sl. No.\tName (Reg. ID) - Status")
	i = 1;
	for x in myresult:
		sql = "SELECT Name FROM users WHERE RegID = %s"
		#print(x)
		val = (x[0], )
		mycursor.execute(sql, val)
		print(str(i) + ".\t" + mycursor.fetchone()[0] + "(" + x[0] + ") - Present")
		i = i + 1

while 1 == 1:
	_ = system('cls')
	
	print("\t\t      Welcome to the")
	print("\t\t   Realtime Attendance")
	print("\t\t Using Face Recognition")
	
	print("\n")
	print("\t1. Register New User/Update Existing User")
	d = datetime.datetime.now()
	cd = d.strftime("%d") + "/" + d.strftime("%m") + "/" + d.strftime("%Y")
	print("\t2. Attendance Report (Today -- " + cd + ")")
	# print("\t3. Attendance Report (Current Month -- " + d.strftime("%B") + ")")
	print("\t3. Enter anything to exit")
	print("\n")
	# Ask for choice i.e. 1, 2, 3
	choice = input('\n Enter your CHOICE and press <return> ==>  ')
	
	if choice == "1":
		# For each person, enter one numeric face id
		face_id = input('\n Enter Registration Number and press <return> ==>  ')
		face_name = input('\n Enter Name and press <return> ==>  ')
		faceDataset(face_id)
		faceTraining()
		insertUserDatabase(face_id,face_name)
		ret_pro = input('\n Registration Complete... Press <return> to continue ==>  ')
		
	elif choice == "2":
		reportToday(cd)
		ret_pro = input('\n Press <return> to continue ==>  ')
	else:
		print("Exiting!!")
		break
