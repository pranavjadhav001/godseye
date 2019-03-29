# god's eye
An facial recognition app<br />
A gui implemenation of face recognition app.
The app is created using Python's Kivy.<br />
based on @ageitgey face recognition:
https://github.com/ageitgey/face_recognition<br />
Also, Take a look at Adam ageitgey's medium brilliant article for further insight link:https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

## Prerequisites
- Tensorflow == 1.12
- Dlib == 19.8.1
- sklearn == 0.20.2
- kivy == 1.10.1
- openface
- OpenCv == 4.0.0
- sqlite3
- numpy
- Download dlib shape predictor dat file and place it in impstuff folder
- Download facenet pretrained model and place it in impstuff folder
link:https://github.com/davidsandberg/facenet
## How to use
#### Run
git clone this repo and download dlib's shape predictor and facenet model and then open 'godseye.py'and change directory path and run.
###### Screen 1 
![alt text](https://github.com/pranavjadhav001/godseye/blob/master/readmeimages/image1.png)__

###### Screen 2 
![alt text](https://github.com/pranavjadhav001/godseye/blob/master/readmeimages/image2.png)<br />
Add profile if you are new.

###### Screen 3
![alt text](https://github.com/pranavjadhav001/godseye/blob/master/readmeimages/image3.png)<br />
Enter details and hit 'submit' and then'start face capture'

###### Screen 4
![alt text](https://github.com/pranavjadhav001/godseye/blob/master/readmeimages/image4.png)<br />
If you have video of the profile, choose 'Video initiate' else choose webcam.

###### Screen 5 
![alt text](https://github.com/pranavjadhav001/godseye/blob/master/readmeimages/image5.png)<br />
Webcam takes your images for training.

###### Results
![alt text](https://github.com/pranavjadhav001/godseye/blob/master/readmeimages/results.png)

## How It works
1. Add profile option allow users to add profiles which get stored in sqlite database
2. Start Face capture is responsible for taking images either from webcam or video.
3. These images get stored in ./images/profile 
4. 'Initiate training' takes all the images of all profiles and get their 128 face embeddings using facenet pre-trained model and classifies them using SVM and saves .pkl file of the svm model.
5. During prediction i.e. "Initiate Face Recognition" , it loads svm model and predicts the current face detected among the given profiles.
6. Dlib is used face detection and calculating facial landmards while openface is responsibile for aligning the face to the centre.
