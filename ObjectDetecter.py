<<<<<<< HEAD
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
 
url='http://192.168.10.162/cam-hi.jpg'
im=None

cap = cv2.VideoCapture(0)
 
def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
 
        cv2.imshow('live transmission',im)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
        
def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:

        success, img = cap.read()
        #img_resp=urllib.request.urlopen(url)
        #imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        #im = cv2.imdecode(imgnp,-1)
 
        bbox, label, conf = cv.detect_common_objects(img)
        img = draw_bbox(img, bbox, label, conf)
 
        cv2.imshow('detection',img)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
 
 
 
#if __name__ == '__main__':
#   print("started")
#    with concurrent.futures.ProcessPoolExecutor() as executer:
           # f1= executer.submit(run1)
#           f2= executer.submit(run2)

cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
while True:

        success, img = cap.read()
        #img_resp=urllib.request.urlopen(url)
        #imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        #im = cv2.imdecode(imgnp,-1)
 
        bbox, label, conf = cv.detect_common_objects(img)
        img = draw_bbox(img, bbox, label, conf)
 
        cv2.imshow('detection',img)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
            
=======
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
 
url='http://192.168.10.162/cam-hi.jpg'
im=None

cap = cv2.VideoCapture(0)
 
def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
 
        cv2.imshow('live transmission',im)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
        
def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:

        success, img = cap.read()
        #img_resp=urllib.request.urlopen(url)
        #imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        #im = cv2.imdecode(imgnp,-1)
 
        bbox, label, conf = cv.detect_common_objects(img)
        img = draw_bbox(img, bbox, label, conf)
 
        cv2.imshow('detection',img)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
 
 
 
#if __name__ == '__main__':
#   print("started")
#    with concurrent.futures.ProcessPoolExecutor() as executer:
           # f1= executer.submit(run1)
#           f2= executer.submit(run2)

cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
while True:

        success, img = cap.read()
        #img_resp=urllib.request.urlopen(url)
        #imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        #im = cv2.imdecode(imgnp,-1)
 
        bbox, label, conf = cv.detect_common_objects(img)
        img = draw_bbox(img, bbox, label, conf)
 
        cv2.imshow('detection',img)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
            
>>>>>>> 0bac90178b1ff25dbdb118fdcd17d74298d4678c
cv2.destroyAllWindows()