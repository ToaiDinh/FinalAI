
from random import sample
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter 
from threading import Thread

import cv2
import PIL.Image,PIL.ImageTk
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import load_img,img_to_array

list_name = ["ChiPu","DieuNhi","DoKhanhVan","HoaiLinh","HoNgocHa","HuynhPhuong","LuongTheThanh","MiDu","MinhHang","NgoThanhVan","NhaPhuong","TangThanhHa","TranThanh","TruongGiang","VanTrang"]

model_architecture = "face_compare_config.json"
model_weights = "face_compare_weights.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
optim = RMSprop()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=20)

isOn = 0
isCompare = 0

def onCamera():
    global isOn
    isOn=1

def onCompare():
    global isCompare
    isCompare = 1

def compare():
    global sample
    global frame
    global actor
    global vn_actor
    global pos
    global canvas,photo
    sample = frame
    img = sample[140:340,220:420]
    img = cv2.resize(img,(128,128))
    img = img.reshape(1,128,128,3)
    img = img.astype('float32')
    img = img/255.0
    pos = int(np.argmax(model.predict(img)))
    patio = np.max(model.predict(img))*100
    patio = round(patio,4)
    path = 'Sample/sample'+str(pos)+'.jpg'
    vn_actor = cv2.imread(path)
    vn_actor = cv2.resize(vn_actor,(350,480))
    cv2.putText(vn_actor,list_name[pos],(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA,False)
    cv2.putText(vn_actor,str(patio)+" %",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA,False)
    vn_actor = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(vn_actor))
    actor.create_image(0,0,image=vn_actor,anchor=tkinter.NW)
    

window = Tk()
window.title('Face compare application')
window.geometry("1060x560")
video = cv2.VideoCapture(1)
canvas_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
canvas_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
canvas = Canvas(window,width=canvas_w,height=canvas_h,bg='blue')
canvas.place(x=10,y=40)
actor = Canvas(window,width=350,height=480,bg='yellow')
actor.place(x=680,y=40)
label1 = Label(text='FROM CAMERA') #- Move your face into red zone')
label1.place(x=10,y=10)
label2 = Label(text="VIETNAMESE ACTOR")
label2.place(x=680,y=10)
btnOn = Button(window,text="ON CAMERA",command=onCamera)
btnOn.place(x=10,y=530)
btnCompare = Button(window,text="COMPARE",command=onCompare)
btnCompare.place(x=680,y=530)


def update_frame():
    global canvas,photo,isCompare
    global actor, vn_actor
    global ret,frame 
    ret,frame = video.read()
    if isOn == 1:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if isCompare == 1:
            thread = Thread(target=compare)
            thread.start()
        #cv2.rectangle(frame,(220,140),(420,340),(255,0,0),2)
        roi = frame
        #Lay "roi" resize theo kich thuoc anh training roi dung de predict
        mask = object_detector.apply(roi)
        contours, _ =cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if(area > 15000):
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
        global pos
        #cv2.putText(frame,"OK",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA,False)
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        canvas.create_image(0,0,image=photo,anchor=tkinter.NW)
    isCompare = 0
    window.after(100,update_frame)

update_frame()
window.mainloop()

