from tkinter import *
from gtts import gTTS
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
import os
import numpy as np
import matplotlib.pyplot as plt
import webbrowser


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

