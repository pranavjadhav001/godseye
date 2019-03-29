from tkinter import *
from tkinter import filedialog as fd
import pickle
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import random
from sklearn.svm import SVC
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import dlib
import cv2
import openface
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.properties import ObjectProperty , StringProperty, ListProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, WipeTransition
from kivy.uix.screenmanager import FadeTransition 
import sys
from kivy.uix.textinput import TextInput
import sqlite3
from kivy.graphics import Color,Rectangle
from kivy.core.audio import SoundLoader
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.popup import Popup
import datetime

os.chdir('path_to_godseye')#E:/godseye
sound_file = './impstuff/IMF.wav'
predictor_model = "./impstuff/shape_predictor_68_face_landmarks.dat"
pkl_filename = "./impstuff/godseye.pkl"
meta_graph = './impstuff/inception model/20170512-110547/model-20170512-110547.meta'
sess_path = './impstuff/inception model/20170512-110547/model-20170512-110547.ckpt-250000'
database_path = './impstuff/minor_project.db'
img_path =  '.\images'

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (0,0,255)
lineType               = 1

def prewhiten(x):
            mean = np.mean(x)
            std = np.std(x)
            std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
            y = np.multiply(np.subtract(x, mean), 1/std_adj)
            return y
#first_screen_of_app
class First(Screen):
    global s
    s=SoundLoader.load(sound_file)
    s.play()
    def sound(self):
        s.stop()
    def done(self):
        App.get_running_app().stop()
        Window.close()

#second screen of app
class Second(Screen):
    def start(self):

        cam = cv2.VideoCapture(0)
        conn = sqlite3.connect(database_path)
        c = conn.cursor()
        c.execute('select * from mission')
        output = c.fetchall()
        my_dict = {}
        for i in output:
            my_dict[i[0]] = i[2]
        print(my_dict)
        if len(my_dict.keys()) > 1:
            with open(pkl_filename, 'rb') as file:  
                pickle_model = pickle.load(file)
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    saver = tf.train.import_meta_graph(meta_graph)
                    saver.restore(tf.get_default_session(), sess_path)
                    while True:
                        ret,image = cam.read()
                        detected_faces = face_detector(image, 1)
                        for i, face_rect in enumerate(detected_faces):
                            pose_landmarks = face_pose_predictor(image, face_rect)

                    # Use openface to calculate and perform the face alignment
                            alignedFace = face_aligner.align(160, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                            alignedFace = prewhiten(alignedFace)
                            img1 = cv2.rectangle(image,(face_rect.left(),face_rect.bottom()),(face_rect.right(),face_rect.top()),(255,0,0),2)
                        
                            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                            alignedFace = alignedFace.reshape(-1,160,160,3)
                            output = sess.run(embeddings,feed_dict={images_placeholder:alignedFace,phase_train_placeholder: False})
                            Ypredict = pickle_model.predict(output)
                            print(Ypredict)
                            print(my_dict[Ypredict[0]])
                            profile_name = my_dict[Ypredict[0]]
                            cv2.putText(image,profile_name, 
                                    (face_rect.left(),face_rect.bottom()), 
                                    font, 
                                    fontScale,
                                    fontColor,
                                    lineType)

                        cv2.imshow("img",image)
                        if(cv2.waitKey(1) == ord('q')):
                            break
                    cam.release()
                    cv2.destroyAllWindows()
        else:
            while True:
                profile_name = my_dict[1]
                ret,image = cam.read()
                if type(image).__name__ != 'NoneType':
                    detected_faces = face_detector(image, 1)
                    for i, face_rect in enumerate(detected_faces):
                         img1 = cv2.rectangle(image,(face_rect.left(),face_rect.bottom()),(face_rect.right(),face_rect.top()),(255,0,0),2)
                         cv2.putText(image,profile_name, 
                                    (face_rect.left(),face_rect.bottom()), 
                                    font, 
                                    fontScale,
                                    fontColor,
                                    lineType)

                    cv2.imshow("img",image)
                    if(cv2.waitKey(1) == ord('q')):
                     break
            cam.release()
            cv2.destroyAllWindows()
        
#third screen of app
class Third(Screen):
    info=ListProperty([])
    txt = StringProperty('')
    tt=ObjectProperty()
    ttt=ObjectProperty()
    tttt=ObjectProperty()
    def profile(self,n,r,c):
        self.info.append(n)
        self.info.append(r)
        self.info.append(c)
        self.txt = n
        lt=[]
        lt.append(n)
        lt.append(r)
        lt.append(c)
        self.data_entry(lt)
        
    def pop(self):
        n=BoxLayout(orientation='vertical',padding=10,spacing=10)
        box=Button(text='Close',font_size=25,size_hint=(0.5,0.4),pos_hint={'center_x':0.5})
        lb=Label(text='Profile Added',font_size=25)
        popup = Popup(title='Finish', auto_dismiss=False,content=n, size_hint=(None, None), size=(400, 400))
        n.add_widget(lb)
        n.add_widget(box)
        box.bind(on_press=popup.dismiss)
        def clr(self):
            self.tt.text=''
            self.ttt.text=''
            self.tttt.text=''
        box.bind(on_release=lambda x:clr(self))
        popup.open()

    def data_entry(self,detail_list):
        conn = sqlite3.connect(database_path)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS mission(id INTEGER PRIMARY KEY AUTOINCREMENT,datestamp TEXT,name TEXT,dob INT,roll_number INT,age INT)")
        insert_data = '''INSERT INTO mission(datestamp,name,dob,roll_number,age)VALUES(?,?,?,?,?)'''
        age = int(datetime.datetime.now().year) - int(detail_list[1][:4])
        print(age)
        c.execute(insert_data,[str(datetime.datetime.now()),detail_list[0],int(detail_list[1]),int(detail_list[2]),age])
        conn.commit()
        c.close()
        conn.close()
#fourth screen of app
class Fourth(Screen):
    inn=ListProperty([])
    got=StringProperty('')
    def webcam(self,detail_list):
        name = detail_list[0]
        
        cam = cv2.VideoCapture(0)
        sample_num = 0
        
        os.mkdir(os.path.join(img_path,name))
        while True:
            ret,image = cam.read()
            detected_faces = face_detector(image,1)
   
            for i,face_rect in enumerate(detected_faces):
                if (face_rect.left()> 0 and face_rect.bottom()> 0 and face_rect.right()> 0 and face_rect.top()> 0):

                    sample_num += 1
                    
                    pose_landmarks = face_pose_predictor(image, face_rect)
                    alignedFace = face_aligner.align(160, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                    cv2.imwrite(os.path.join(os.path.join(img_path,name),name)+ "." +str(sample_num)+".jpg",alignedFace)
                    img1 = cv2.rectangle(image,(face_rect.left(),face_rect.bottom()),(face_rect.right(),face_rect.top()),(255,0,0),2)
                    cv2.waitKey(100)

            cv2.imshow("img",image)
            if(cv2.waitKey(1) == ord('q')) or sample_num > 50:
                break
        cam.release()
        cv2.destroyAllWindows()

    def video_finder(self,detail_list):
        root  = Tk()
        root.withdraw()
        video_path = fd.askopenfilename()
        name = detail_list[0]
        cam = cv2.VideoCapture(video_path)
        sample_num = 0
        os.mkdir(os.path.join(img_path,name))

        while True:
            ret,image = cam.read()
            detected_faces = face_detector(image,1)

            for i,face_rect in enumerate(detected_faces):
                if (face_rect.left()> 0 and face_rect.bottom()> 0 and face_rect.right()> 0 and face_rect.top()> 0):

                    sample_num += 1
                    
                    pose_landmarks = face_pose_predictor(image, face_rect)
                    alignedFace = face_aligner.align(160, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                    ccv2.imwrite(os.path.join(os.path.join(img_path,name),name)+ "." +str(sample_num)+".jpg",alignedFace)
                    img1 = cv2.rectangle(image,(face_rect.left(),face_rect.bottom()),(face_rect.right(),face_rect.top()),(255,0,0),2)
                    cv2.waitKey(100)

            cv2.imshow("img",image)
            if(cv2.waitKey(1) == ord('q')) or sample_num > 50:
                break
        cam.release()
#fifth screen of app
class Fifth(Screen):
    def train(self):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(meta_graph)
                saver.restore(tf.get_default_session(), sess_path)
                final_outputs = []
                label_outputs = []

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_dict = {}
                for subdir, dirs, files in os.walk(img_path):
                    for file in files:
                        print(file)
                        path = os.path.join(subdir, file)
                        image = cv2.imread(path)
                        image = prewhiten(image)
                        image = np.array(image).reshape(-1,160,160,3)
                        output = sess.run(embeddings,feed_dict={images_placeholder:image,phase_train_placeholder: False})
                        final_outputs.append(output[0])
                        label_output = os.path.basename(file).split('.')[0]
                        
                        
                        label_outputs.append(label_output)
        conn = sqlite3.connect(database_path)
        c = conn.cursor()
        c.execute('select * from mission')
        output = c.fetchall()
        my_dict = {}
        for i in output:
            my_dict[i[2]] = i[0]
        labele_outputs = []
        for i in label_outputs:
            labele_outputs.append(my_dict[i])
            
        c = list(zip(final_outputs,labele_outputs))
        random.shuffle(c)
        a,b = zip(*c)
        if len(np.unique(np.array(b))) > 1:
            print("training started")
            model = SVC(kernel='linear', probability=True)
            model.fit(a, b)
            pickle.dump(model, open(pkl_filename, 'wb'))
        self.manager.current='second'

            

class Manager(ScreenManager):
    pass

root=Builder.load_string('''
#: import WipeTransition kivy.uix.screenmanager.WipeTransition
Manager:
    transition: WipeTransition(duration=2)
    First:
    Second:
    Third:
        id:fast
    Fourth:
        got:fast.txt
        inn:fast.info
    Fifth:    
<First>:
    name:'first'
    BoxLayout:
        orientation: 'vertical'
        Image:
            source: r'E:\godseye\impstuff\Eye.jpg'
            size_hint:None,None
            width:'500'
            height:'500'
            pos_hint:{'center_x':0.5}
        BoxLayout:
            padding:100
            spacing:100
            Button:
                text:'ENTER'
                font_size:25
                size_hint: 0.5, None
                height:"85dp"
                pos_hint:{'y':0.2}
                on_press: app.root.current='second'
                
            Button:
                text:'EXIT'
                font_size:25
                size_hint: 0.5,None
                height:"85dp"
                pos_hint:{'y':0.2}
                on_press: root.done()
        Label:
            text:'WELCOME'
            font_size:30
            pos_hint:{'center_x':0.5}
<Second>:
    name:'second'
    BoxLayout:
        orientation: 'vertical'
        padding:30
        spacing:100
        Button:
            text:'Initiate Face Recognition'
            font_size:25
            pos_hint:{'center_x':0.5}
            size_hint_x:0.4
            on_press: root.start()
        Button:
            text: 'Add Profile'
            font_size:25
            pos_hint:{'center_x':0.5}
            size_hint_x:0.4
            on_press: app.root.current='third'
        Button:
            text: 'Back'
            font_size:25
            pos_hint:{'center_x':0.5}
            size_hint_x:0.4
            on_press: app.root.current='first'
<Third>:
    tt:name
    ttt:dob
    tttt:rollno
    name:'third'
    BoxLayout:
        orientation: 'vertical'
        padding:20
        spacing:10
        BoxLayout:
            Label:
                text:'Name'
                font_size:25
                size_hint_x:None
                width:'175'
            TextInput:
                id:name
                multiline:False
                size_hint_y:None
                height:'50'
                pos_hint:{'y':0.35}
        BoxLayout:
            Label:
                text:'DOB(yyyymmdd)'
                font_size:25
                size_hint_x:None
                width:'175'
            TextInput:
                id: dob
                multiline:False
                size_hint_y:None
                height:'50'
                pos_hint:{'y':0.35}
        BoxLayout:
            Label:
                text:'Roll Number'
                font_size:25
                size_hint_x:None
                width:'175'
            TextInput:
                id: rollno
                multiline:False
                size_hint_y:None
                height:'50'
                pos_hint:{'y':0.35}
        BoxLayout:
            
            spacing:50
            size_hint_y:None
            height:'60'
            Button:
                text:"Back"
                font_size:25
                size_hint_x:0.3
                on_press:app.root.current='second'
            Button:
                text:'Start Face Capture'
                font_size:25
                size_hint_x:0.3
                on_press:app.root.current='fourth'
            Button:
                text:"Submit"
                font_size:25
                size_hint_x:0.3
                on_press:root.profile(name.text,dob.text,rollno.text)
                on_release: root.pop()

<Fourth>:
    name:'fourth'
    BoxLayout:
        orientation:'vertical'
        padding:50
        spacing:40
        Label:
            text:'Launch'
            font_size:35
            size_hint_y:0.4
        BoxLayout:
            
            Button:
                text:'WebCam Initiate'
                font_size:25
                size_hint_y:0.3
                size_hint_x:0.3
                pos_hint:{'center_y':0.7}
                on_press:root.webcam(root.inn)
            Label:
                text:'OR'
                font_size:35
                size_hint_y:0.3
                size_hint_x:0.3
                pos_hint:{'center_y':0.7}
            Button:
                text:'Video Initiate'
                font_size:25
                size_hint_y:0.3
                size_hint_x:0.3
                pos_hint:{'center_y':0.7}
                on_press:root.video_finder(root.inn)
        Button:
            text:'Initiate Training'
            font_size:25
            pos_hint:{'center_x':0.5}
            size_hint_x:0.4
            size_hint_y:0.3
            on_press:app.root.current='fifth'

<Fifth>:
    name:'fifth'
    on_enter:root.train()
    Label:
        text:"PLEASE WAIT!!!!"

''')


class Crank(App):
    def build(self):
        return root
Crank().run()
    
