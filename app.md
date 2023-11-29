```
import cv2
import tensorflow as tf
import numpy as np
```
#### We begin by importing necessary libraries. cv2 is OpenCV, a library for computer vision tasks, and numpy is a library for numerical operations. tensorflow is used to load the trained model.

```
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

model = tf.keras.models.load_model('final_model_weights.hdf5')
```
#### You define the class_names list to map predicted class indices to human-readable emotions. Then, you load your pre-trained model using TensorFlow's Keras API.

```
video = cv2.VideoCapture(0)
```
#### Here, you initiate a connection to your computer's webcam using OpenCV's VideoCapture function.

```
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

#### You initialize a face detection classifier using OpenCV's Haar Cascade classifier, which is a pre-trained model specifically designed for detecting faces.

```
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
```
#### This is the main loop where you read frames from the webcam feed. You convert each frame to grayscale, which is typically what face detection algorithms use. Then, you use the detectMultiScale function to find faces in the grayscale frame.

```
    for x, y, w, h in faces:
        sub_face_img = gray[y : y + h, x : x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
```

#### For each detected face, you isolate the face region, resize it to match your model's input size, and normalize its pixel values. You reshape the normalized image to fit the model's expected input shape. You then use the trained model to predict the emotion label for the face.

```
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, class_names[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
```

#### In this block, you draw rectangles around the detected faces and place a label above each face with the predicted emotion. This adds a visual indication of the detected emotions on the webcam feed.

```
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
```
#### Here, you display the modified frame in a window named "Frame". You use cv2.waitKey to wait for user input. If the user presses the 'q' key, the loop will break, and the program will exit.

```
video.release()
cv2.destroyAllWindows()
```
#### Finally, after exiting the loop, you release the webcam and close any open windows created by OpenCV.