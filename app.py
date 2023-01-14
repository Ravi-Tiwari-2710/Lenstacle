from flask import Flask, render_template, Response
import cv2 , os
import face_recognition
import numpy as np
import pyttsx3
import cvlib as cv
from cvlib.object_detection import draw_bbox

#from flask_ngrok import run_with_ngrok



app=Flask(__name__)
#run_with_ngrok(app)
camera = cv2.VideoCapture(0)
detected_obj=[]


path = 'image_folder'
url='http://192.168.231.162/cam-hi.jpg'
##'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''
 

    
 
images = []
classNames = []
myList = os.listdir(path)
#print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
 

 
 
encodeListKnown = findEncodings(images)
#print('Encoding Complete')




    






def gen_frames():


    def Speak(text):
        engine = pyttsx3.init("sapi5")
        voices = engine.getProperty('voices') 
        engine.setProperty('voices', voices[2].id)
        engine.setProperty('rate', 180)
        engine.say(text)
        engine.runAndWait()
    
    while True:
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            bbox, label, conf = cv.detect_common_objects(imgS)
            

            if label ==['person']:
                facesCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
 
                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
                    matchIndex = np.argmin(faceDis)
 
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
# print(name)
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        obj = name
                        if obj not in detected_obj:
                            Speak(f"{obj} is near you...")
                            detected_obj.append(obj)

                    else:
                        obj = 'unknown person'
                        if obj not in detected_obj:
                            Speak(f"{obj} is near you...")
                            detected_obj.append(obj)


            else:
                img = draw_bbox(imgS, bbox, label, conf)
                obj = label
                if obj not in detected_obj and len(obj)>0:
                    Speak(f"{obj} is detected near you....")
                    detected_obj.append(obj)


            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')







@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run()
