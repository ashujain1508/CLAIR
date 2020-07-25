import tkinter as tk
from tkinter import *
import requests, json 

import faceRecognition as fr
from gtts import gTTS
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
import os
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import pygame
import os


import tensorflow as tf
import cv2
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np

from PIL import *

window=tk.Tk()
window.title("Corona Tool-Kit")


l1 = tk.Label(window, text="--:Welcome To Custodian:--", font=('Comic Sans MS',30), bg='#4bceb3')
l1.grid(padx=0, pady=0, sticky=N)



corona = """Hey, Welcome to the 1st GUI of your custodian As the Name suggest my 
 responsablity is to take care of you its my responsablity.I will do that by keeping
you up to date with information related to deadly virus and i will make sure that you
are wearing your mask before going somewhere or not i also  have some mood changing 
music  just click the buttons according to your choice for   exploring me!!!!!! """


l3 = tk.Label(window, text=corona, font=('Comic Sans MS',17),bg='#2eb8b8' ,fg='darkred')
l3.grid(padx=200, pady=50, sticky=N)

l2 = tk.Label(window,text="Click Detect With ME To Know If You Are Missing Something Or Not  ", font=('Comic Sans MS',20,),bg='#4bceb3' ,fg='DarkRed')
l2.grid(padx=300, pady=90, sticky=N)


def mask():
    
    model=tf.keras.models.load_model('Mask_detector_model.h5')

    #loading the cascades
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #webcam face recognition
    video_capture=cv2.VideoCapture(0)
    while True:
        _,frame=video_capture.read()
        faces=face_cascade.detectMultiScale(frame,1.3,5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            face=frame[y:y+h,x:x+w]
            cropped_face=face
        
            if type(face) is np.ndarray:
                face=cv2.resize(face,(224,224))
                im=Image.fromarray(face,'RGB')
                img_array=np.array(im)
                img_array=np.expand_dims(img_array,axis=0)
                pred=model.predict(img_array)
                print(pred)
                
                if(pred[0][0]>0.6):
                    prediction='Mask Found'
                    cv2.putText(cropped_face,prediction,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                else:
                    prediction='No Mask'
                    cv2.putText(cropped_face,prediction,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(frame,'No Face Found',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                
        cv2.imshow('Video',frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


b1=tk.Button(window,text="Detect With ME!!",font=("Algerian",15), bg='darkred',fg='white', command=mask).grid(padx=0, pady=0, sticky=E)

def index():
    
    window= Tk()
    window.title("corona cases")
    window.config(bg = "#828481")


    def data():
        extract_contents = lambda row: [x.text.replace('\n', '') for x in row]
        URL = 'https://www.mohfw.gov.in/'

        SHORT_HEADERS = ['SNo', 'State', 'Indian-Confirmed',
                        'Foreign-Confirmed', 'Cured', 'Death']

        response = requests.get(URL).content
        soup = BeautifulSoup(response, 'html.parser')
        header = extract_contents(soup.tr.find_all('th'))

        stats = []
        all_rows = soup.find_all('tr')

        for row in all_rows:
            stat = extract_contents(row.find_all('td'))
            if stat:
                if len(stat) == 5:
                    # last row
                    stat = ['', *stat]
                    stats.append(stat)
                elif len(stat) == 6:
                    stats.append(stat)

        stats[-1][1] = "Total Cases"

        stats.remove(stats[-1])
        objects = []
        for row in stats :
            objects.append(row[1])

        y_pos = np.arange(len(objects))

        performance = []
        for row in stats :
            performance.append(row[2] + str(int(float(row[3]))))

        table = tabulate(stats, headers=SHORT_HEADERS)
        print(table)



    def audio():
        my_text = "Always protect your face with a handkerchief or tissue while coughing or sneezing." \
                " Regularly clean hands with soap. Avoid touching your face, eyes, or nose." \
                " If someone has cough, fever, or breathlessness maintain one metre distance" \
                ". If needed, visit your nearest health centre immediately"
        language = 'en'
        myobj = gTTS(text=my_text, lang=language, slow=False)
        myobj.save("welcome.mp3")
        os.system("welcome.mp3")




    def quiz():
        score =0
        print("How long does the novel coronavirus survive outside the body?")
        print("option 1:A week in the air and on surface"
            "option 2: several hours to days "
            "option 3: upto 2 and half week")
        a=int(input("enter the answer 1 2 or 3"))
        if(a==2):
            print("correct answer ")
            score +=1
        else:
            print("incorect option "
                "The virus that causes Covid-19 is stable for several hours in tiny floating particles, "
                "known as aerosols, and up to two to three days on plastic and stainless steel, U.S. researchers showed."
                " A study published in the New England Journal of Medicine in March concluded that the novel coronavirus "
                "can survive four hours on copper, 24 hours on cardboard, 48 hours on stainless steel and 72 hours on plastic.")

        print("What’s more important for preventing infection?")
        print("option 1: frequently hand wasing "
            "option 2: wearing a face mask ")
        b=int(input("enter the answer 1 or 2 "))
        if(b==1):
            print("correct answer ")
            score +=1
        else:
            print(" wrong answer ")




    def check ():
        global check_screen
        check_screen = Toplevel(window)
        check_screen.title("check")
        check_screen.geometry("300x250")
        global name
        global address
        global number
        global symptom
        global name_entry
        global address_entry
        global number_entry
        global symptom_entry
        name= StringVar()
        address= StringVar()
        number= StringVar()
        symptom = StringVar()
        info = Label(check_screen, text="PLEASE ENTER YOUR NAME", bg="#C8F9C4").pack()

        name_entry = Entry(check_screen, textvariable=name)
        name_entry.pack()
        info1 = Label(check_screen, text="ENTER YOUR ADDRESS", bg="#C8F9C4").pack()
        address_entry = Entry(check_screen, textvariable=address)
        address_entry.pack()
        info2 = Label(check_screen, text="ENTER YOUR MOBILE NUMBER", bg="#C8F9C4").pack()
        number_entry=Entry(check_screen, textvariable=number)
        number_entry.pack()
        info3 = Label(check_screen, text="SPECIFY YOUR SYMPTOM ", bg="#C8F9C4").pack()
        symptom_entry = Entry(check_screen, textvariable=symptom)
        symptom_entry.pack()
        Button(check_screen, text="enter", width=10, height=1, command=symptom_check).pack()




    def symptom_check ():
        username_info=name.get()
        address_info=address.get()
        number_info = number.get()
        symptom_info= symptom.get()
        file = open(username_info, "w")
        file.write(username_info + "\n")
        file.write(address_info + "\n")
        file.write(number_info + "\n")
        file.write(symptom_info)
        file.close()
        name_entry.delete(0,END)
        address_entry.delete(0,END)
        number_entry.delete(0,END)
        symptom_entry.delete(0,END)
        l6=Label(check_screen, text="THANKS FOR THE DATA", fg="green", font=("calibri", 11)).pack()





    explan_corona = """
    Coronaviruses are a group of related RNA viruses that cause diseases in mammals and birds.
    In humans, these viruses cause respiratory tract infections that can range from mild to lethal.
    Mild illnesses include some cases of the common cold (which is caused also by certain other viruses, predominantly 
    rhinoviruses),while more lethal varieties can cause SARS, MERS, and COVID-19. Symptoms in other species vary:
    in chickens, they cause an upper respiratory tract disease, while in cows and pigs they cause diarrhea. 
    There are as yet no vaccines or antiviral drugs to prevent or treat human coronavirus infections.
    Coronaviruses constitute the subfamily Orthocoronavirinae, in the family Coronaviridae, order Nidovirales,
    and realm Riboviria. They are enveloped viruses with a positive-sense single-stranded RNA genome and 
    a nucleocapsid of helical symmetry. This is wrapped in a icosahedral protein shell. The genome size of 
    coronaviruses ranges from approximately 26 to 32 kilobases, one of the largest among RNA viruses. They have
    characteristic club-shaped spikes that project from their surface, which in electron micrographs create an image
    reminiscent of the solar corona, from which their name derives.."""





    symptom = """Common symptoms:
    fever.
    tiredness.
    dry cough.

    Some people may experience:
    aches and pain.
    nasal congestion.
    runny nose.
    sore throat.
    diarrhoea."""




    leftFrame = Frame(window, width=200, height = 600, bg="#C8F9C4", highlightthickness=2, highlightbackground="#111")
    leftFrame.grid(row=0, column=0, padx=10, pady=2, sticky=N+S)

    case = Label(leftFrame, text="TO KNOW CASES IN INDIA CLICK THE BUTTON GIVEN BELOW:",  anchor=W, bg="#C8F9C4")
    case.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    Btn_case = Button(leftFrame, text="click", command=data, bg="#EC6E6E")
    Btn_case.grid(row=1, column=0, padx=10, pady=2)
    info= Label(leftFrame, text="IMPORTANT INFORMATON FOR CORONA VIRUS ", anchor=W, bg="#C8F9C4")
    info.grid(row=4, column=0, padx=10, pady=2, sticky=W)
    info1 = Label(leftFrame, text=explan_corona, anchor=W, bg="#C8F9C4")
    info1.grid(row=5, column=0, padx=10, pady=2, sticky=W)
    link1 = Label(window, text="FOR MORE INFO CLICK", fg="blue", cursor="hand2")
    link1.grid()
    link1.bind("<Button-1>", lambda e: callback("file:///C:/Users/Hp/Desktop/hackathon/hackathon1.html"))






    rightFrame = Frame(window, width=200, height = 600, bg="#C8F9C4", highlightthickness=2, highlightbackground="#111")
    rightFrame.grid(row=0, column=1, padx=10, pady=2, sticky=N+S)
    rl1 = Label(rightFrame, text="symptoms", anchor=S, justify=LEFT, bg="#C8F9C4")
    rl1.grid(row=0, column=0, padx=10, pady=2)
    info1 = Label(rightFrame, text=symptom, anchor=N, bg="#C8F9C4")
    info1.grid(row=1, column=0, padx=10, pady=2, sticky=S)
    info2 = Label(rightFrame, text="AUDIO FILE ABOUT CORONA", anchor=N, bg="#C8F9C4")
    info2.grid(row=2, column=0, padx=10, pady=2, sticky=S)
    audioBtn = Button(rightFrame, text="audio", command=audio, bg="#EC6E6E")
    audioBtn.grid(row=3, column=0, padx=10, pady=2)
    quizBtn = Button(rightFrame, text="quiz", command=quiz, bg="#EC6E6E")
    quizBtn.grid(row=4, column=0, padx=10, pady=2)
    info3 = Label(rightFrame, text="IF YOU ARE FACING ANY SYMPTOM CLICK DOWN", anchor=N, bg="#C8F9C4")
    info3.grid(row=5, column=0, padx=10, pady=2, sticky=S)
    checkBtn = Button(rightFrame, text="CHECK", command=check, bg="#EC6E6E")
    checkBtn.grid(row=6, column=0, padx=10, pady=2)


    window.mainloop()





b2=tk.Button(window,text="Let's Explore Details of This Virus",font=("Algerian",15), bg='Dark Red',fg='white', command=index)
b2.grid(padx=0, pady=0, sticky=S+W)

def play():
    class MusicPlayer:

        def __init__(self,root):
            self.root = root
        
            self.root.title("Music Player")
            
            self.root.geometry("1000x200+200+200")
            
            pygame.init()
            
            pygame.mixer.init()
            
            self.track = StringVar()
            
            self.status = StringVar()


            
            trackframe = LabelFrame(self.root,text="Song Track",font=("times new roman",15,"bold"),bg="grey",fg="white",bd=5,relief=GROOVE)
            trackframe.place(x=1,y=0,width=600,height=100)
        
            songtrack = Label(trackframe,textvariable=self.track,width=20,font=("times new roman",24,"bold"),bg="grey",fg="gold").grid(row=0,column=0,padx=10,pady=5)
            
            trackstatus = Label(trackframe,textvariable=self.status,font=("times new roman",24,"bold"),bg="grey",fg="gold").grid(row=0,column=1,padx=10,pady=5)
        
            buttonframe = LabelFrame(self.root,text="Control Panel",font=("times new roman",15,"bold"),bg="grey",fg="white",bd=5,relief=GROOVE)
            buttonframe.place(x=0,y=100,width=600,height=100)
            
            playbtn = Button(buttonframe,text="PLAY",command=self.playsong,width=6,height=1,font=("times new roman",16,"bold"),fg="navyblue",bg="gold").grid(row=0,column=0,padx=10,pady=5)
            
            playbtn = Button(buttonframe,text="PAUSE",command=self.pausesong,width=8,height=1,font=("times new roman",16,"bold"),fg="navyblue",bg="gold").grid(row=0,column=1,padx=10,pady=5)
            
            playbtn = Button(buttonframe,text="UNPAUSE",command=self.unpausesong,width=10,height=1,font=("times new roman",16,"bold"),fg="navyblue",bg="gold").grid(row=0,column=2,padx=10,pady=5)
        
            playbtn = Button(buttonframe,text="STOP",command=self.stopsong,width=6,height=1,font=("times new roman",16,"bold"),fg="navyblue",bg="gold").grid(row=0,column=3,padx=10,pady=5)
            
            songsframe = LabelFrame(self.root,text="Song Playlist",font=("times new roman",15,"bold"),bg="grey",fg="white",bd=5,relief=GROOVE)
            songsframe.place(x=600,y=0,width=400,height=200)
            
            scrol_y = Scrollbar(songsframe,orient=VERTICAL)
        
            self.playlist = Listbox(songsframe,yscrollcommand=scrol_y.set,selectbackground="gold",selectmode=SINGLE,font=("times new roman",12,"bold"),bg="silver",fg="navyblue",bd=5,relief=GROOVE)
            
            scrol_y.pack(side=RIGHT,fill=Y)
            scrol_y.config(command=self.playlist.yview)
            self.playlist.pack(fill=BOTH)
            
            os.chdir("songs")

            songtracks = os.listdir()
        
            for track in songtracks:
                self.playlist.insert(END,track)
        
        def playsong(self):

            self.track.set(self.playlist.get(ACTIVE))
            
            self.status.set("-Playing")
            
            pygame.mixer.music.load(self.playlist.get(ACTIVE))
        
            pygame.mixer.music.play()
        def stopsong(self):
        
            self.status.set("-Stopped")
        
            pygame.mixer.music.stop()
        def pausesong(self):
            
            self.status.set("-Paused")
        
            pygame.mixer.music.pause()
        def unpausesong(self):
        
            self.status.set("-Playing")
            
            pygame.mixer.music.unpause()

    root = Tk()

    MusicPlayer(root)

    root.mainloop()



b3=tk.Button(window,text="Listing To Music",font=("Algerian",15), bg='#111223',fg='white', command=play)
b3.grid(padx=0, pady=50, sticky=S)

def dete():
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





b5=tk.Button(window,text="Conform Your Identity Here",font=("Algerian",15), bg='DarkRed',fg='white', command=dete)
b5.grid(padx=0, pady=60, sticky=S)

def weather():
    api_key = "0caba6521bc817d871458a858410a38c"
  

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    

    city_name = input("Enter city name : ") 
    

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name 
    

    response = requests.get(complete_url) 
    

    x = response.json() 
    

    if x["cod"] != "404": 
    
    
        y = x["main"] 
    
        current_temperature = y["temp"] 
    

        current_pressure = y["pressure"] 
    
        current_humidiy = y["humidity"] 
    
    
        z = x["weather"] 
    
    
        weather_description = z[0]["description"] 
    
    
        print(" Temperature (in kelvin unit) = " +
                        str(current_temperature) + 
            "\n atmospheric pressure (in hPa unit) = " +
                        str(current_pressure) +
            "\n humidity (in percentage) = " +
                        str(current_humidiy) +
            "\n description = " +
                        str(weather_description)) 
    
    else: 
        print(" City Not Found ") 



b4=tk.Button(window,text="Check Weather Of Your City With Consol",font=("Algerian",15), bg='#111223',fg='white', command=weather)
b4.grid(padx=0, pady=50, sticky=S)


l4 = tk.Label(window, text="Wear Mask !!! Stay Protected From CoronaVirus ", font=('Comic Sans MS',25),bg='#4bceb3' ,fg='DarkRed')
l4.grid(padx=0, pady=100, sticky=S)

window.configure(background = '#4bceb3')
window.geometry("1500x950")
window.mainloop()




