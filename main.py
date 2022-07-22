# Facial Detection Using python libraries such as cv2(openCV) and dlib.
# Made By @RonitGandhi

# modules
import cv2       #download this modules using pip install 
import dlib

cap = cv2.VideoCapture(0)  # Making a Variable "cap" which contains our videocaptured from our webcam using cv2 module.

detector = dlib.get_frontal_face_detector() # Making a variable "detector" which contains the face detection using dlib module and a dlib's function {.get_frontal_face_detector()}


predictor = dlib.shape_predictor(
    "c:\\Programming\\Python\\Face Detection\\shape_predictor_68_face_landmarks.dat"
)  	# Making a variable "predictor" which contains the points or dots which are going to be displayed on face. This is a .dat file. It is provide with the source code in my github("https://github.com/ronit18")


while True:    # The main loop.

    _, frame = cap.read()   # this reads the frame caputure from our webcam using cv2 module.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # We give the cv2 module one colour so it can process fast..

    faces = detector(gray)   # here we have given the gray color.

    for face in faces:       # loop for face

        print(faces)
		# This are the coordinates of our face.
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3) # --> uncomment this line... to get a box around your face.

        landmarks = predictor(gray, face)   # this gives us the landmarks of our face(i.e the dots on face) but it only shows on the terminal. so next we use the circle function of cv2 module so it can show the dots on face.
        print(landmarks)

        for n in range(0, 68):    # loop for dots. this will show 67 dots on face.

			# loop will continuously add dots on face by puting values in place of n (0->67) with the help of for loop.
            x = landmarks.part(n).x   
            y = landmarks.part(n).y

			# This is the circle function (.cicle(img,center,radius,color, thickness.)) frame, (x, y), 2, (0, 255, 0), -1) 
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) 
			 # img == frame , center == (x,y), radius == 6, color == (0,255,0) Green, thickness == -1.         

    cv2.imshow("Face Detection made by @RonitGandhi", frame)   # This shows the title on our frame of video.

    key = cv2.waitKey(1) 
							# this is for quiting the loop or closing the frame of video.		
    if key == 27:
        break


