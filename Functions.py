import speech_recognition as sr
from googletrans import Translator
import openai
from dotenv import load_dotenv
import pyttsx3

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
    

def Listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening..........")
        r.pause_threshold = 1
        audio = r.listen(source,0,8)
    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language="hi") #Using google for voice recognition.
    except :    
        return ""
    Translate = Translator()
    result = Translate.translate(query)
    data = result.text
    print(f"you:{data}.")
    return data


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




def TranslateHinditoEng(Text):
    line = str(Text)
    translate = Translator()
    result = translate.translate(line)
    data = result.text
    print(f"you:{data}.")
    return data


def TakeCommand():
    query = Listen()
    data = TranslateHinditoEng(query)
    return data


def Speak(text):
    engine.say(text)
    engine.runAndWait()

while True:

    query = ListenEnglish()
    print(query)
    res = Reply(query)
    print(res)
    Speak(res)


    