<<<<<<< HEAD
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import face_recognition
import os
import speech_recognition as sr
from googletrans import Translator
import openai
from dotenv import load_dotenv
import pyttsx3
import concurrent.futures


openai.api_key = "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"
load_dotenv()

engine = pyttsx3.init("sapi5")
voices = engine.getProperty('voices') 
engine.setProperty('voices', voices[2].id)
engine.setProperty('rate', 180)

def Reply(query, chat_log=None):
    FileLog = open("chat_log.txt", 'r')
    chat_log_templete = FileLog.read()
    FileLog.close()

    if chat_log is  None:
        chat_log = chat_log_templete
    

    prompt = f'{chat_log}Ravi : {query}\nJason : '
    try:
        response = openai.Completion.create(
        model="text-davinci-002",
        prompt= prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[ "Jason:"]
        )
        answer = response.choices[0].text.strip()
        chat_log_templete_update = chat_log_templete + f'\nRavi : {query} \nJason : {answer}'
        FileLog = open("chat_log.txt", "w")
        FileLog.write(chat_log_templete_update)
        FileLog.close()
        return answer
    except:
        Speak("say that again please..")


def ListenEnglish():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening..........")
        r.pause_threshold = 1
        audio = r.listen(source,0,10)
    try:
        print("Recognizing...")    
        query = r.recognize_google(audio) #Using google for voice recognition.

    except : 
        return ""
    return str(query)


def Speak(text):
    engine.say(text)
    engine.runAndWait()

 
url='http://192.168.43.250/capture?_cb=19476'
im=None

cap = cv2.VideoCapture(0)

path = 'image_folder'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
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
print('Encoding Complete')


def run1():
    i = 0
    while i<70:

            success, img = cap.read()
            #img_resp=urllib.request.urlopen(url)
            #imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
            #img = cv2.imdecode(imgnp,-1)
 
            bbox, label, conf = cv.detect_common_objects(img)
            #print(label)
        
        

            if label == ['person']:
                    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

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
                            global na 
                            na = name
                        else:
                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, "unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                            na = "unknown person"
                    #cv2.imshow('detection',img)
            img = draw_bbox(img, bbox, label, conf)
            na = label
           

 
            cv2.imshow('output',img)
            key=cv2.waitKey(1)
            if key==ord('q'):
                break
    i = i+1        
    cv2.destroyAllWindows()
    return na


def Assistant():
    while True:
        query = ListenEnglish()
        print(query)
        res = Reply(query)
        print(res)
        Speak(res)




if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:


            f1= executer.submit(run1)
=======
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import face_recognition
import os
import speech_recognition as sr
from googletrans import Translator
import openai
from dotenv import load_dotenv
import pyttsx3
import concurrent.futures


openai.api_key = "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"
load_dotenv()

engine = pyttsx3.init("sapi5")
voices = engine.getProperty('voices') 
engine.setProperty('voices', voices[2].id)
engine.setProperty('rate', 180)

def Reply(query, chat_log=None):
    FileLog = open("chat_log.txt", 'r')
    chat_log_templete = FileLog.read()
    FileLog.close()

    if chat_log is  None:
        chat_log = chat_log_templete
    

    prompt = f'{chat_log}Ravi : {query}\nJason : '
    try:
        response = openai.Completion.create(
        model="text-davinci-002",
        prompt= prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[ "Jason:"]
        )
        answer = response.choices[0].text.strip()
        chat_log_templete_update = chat_log_templete + f'\nRavi : {query} \nJason : {answer}'
        FileLog = open("chat_log.txt", "w")
        FileLog.write(chat_log_templete_update)
        FileLog.close()
        return answer
    except:
        Speak("say that again please..")


def ListenEnglish():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening..........")
        r.pause_threshold = 1
        audio = r.listen(source,0,10)
    try:
        print("Recognizing...")    
        query = r.recognize_google(audio) #Using google for voice recognition.

    except : 
        return ""
    return str(query)


def Speak(text):
    engine.say(text)
    engine.runAndWait()

 
url='http://192.168.43.250/capture?_cb=19476'
im=None

cap = cv2.VideoCapture(0)

path = 'image_folder'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
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
print('Encoding Complete')


def run1():
    i = 0
    while i<70:

            success, img = cap.read()
            #img_resp=urllib.request.urlopen(url)
            #imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
            #img = cv2.imdecode(imgnp,-1)
 
            bbox, label, conf = cv.detect_common_objects(img)
            #print(label)
        
        

            if label == ['person']:
                    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

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
                            global na 
                            na = name
                        else:
                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, "unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                            na = "unknown person"
                    #cv2.imshow('detection',img)
            img = draw_bbox(img, bbox, label, conf)
            na = label
           

 
            cv2.imshow('output',img)
            key=cv2.waitKey(1)
            if key==ord('q'):
                break
    i = i+1        
    cv2.destroyAllWindows()
    return na


def Assistant():
    while True:
        query = ListenEnglish()
        print(query)
        res = Reply(query)
        print(res)
        Speak(res)




if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:


            f1= executer.submit(run1)
>>>>>>> 0bac90178b1ff25dbdb118fdcd17d74298d4678c
            f2= executer.submit(Assistant)