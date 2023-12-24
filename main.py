# import the opencv library
import cv2
import tensorflow as tf
import torch
import time
import numpy as np

# Detect Model
from ultralytics import YOLO

faceNet=YOLO('FaceDetect/best.pt')

# Age-Gender Model
agegenderModel = tf.keras.models.load_model(r"FaceRecog/pretrained/agegender_Lib50_100epoch.h5")

# Expression model:
from FaceEmote.model import ResNet18
from FaceEmote.predict import predict as emote_predict, load_model
from FaceRecog.predict import predict as age_gender_predict

expression_model = ResNet18()
optimizer = torch.optim.SGD(expression_model.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-4, nesterov=True)

expression_model, optimizer = load_model(expression_model, optimizer, "FaceEmote/pretrained/Express_model_final.pth")



# Get Face Image
def get_face(faceNet, frame, img_resize = (224,224)):
    results=faceNet(frame)
    bboxs=results[0].boxes.xyxy.numpy().astype('int')
    faceLoad=[]
    for box in bboxs:
        x1=box[0]
        y1=box[1]
        x2=box[2]
        y2=box[3]
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 2)
        faceLoad.append(cv2.resize(frame[y1:y2, x1:x2], img_resize))
    return frame, faceLoad, bboxs


# define a video capture object
# vid = cv2.VideoCapture(0)
vid = cv2.VideoCapture('data_test/vu.jpg')
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

while(True):
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)

    if type(frame) != type(None):
        frame, faceLoad, bboxs = get_face(faceNet, frame, img_resize=(200,200))

        # Predict facial expression and print it into frame
        for i in range(len(faceLoad)):
                x = bboxs[i][2] + 10
                y = bboxs[i][1] + 20
                faceImg = faceLoad[i]

                # if int(time.time())%3==0:
                emote_prediction = emote_predict(faceLoad[i], expression_model)
                pred_gender, pred_age, pred_age_cat = age_gender_predict(faceImg, agegenderModel)
                
                cv2.putText(frame, f'Gender: {pred_gender}', (x, y+20*0), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Age: {pred_age}', (x, y+20*1), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, f'Age Group: {pred_age_cat}', (x, y+20*2), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                # cv2.putText(frame, f'Ethnic: {pred_ethnic}', (x, y+20*3), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                
                cv2.putText(frame, f'Mood: {emote_prediction}', (x,y+100+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))


        cv2.imshow('frame',frame)
        # cv2.imwrite('pred_img.jpeg', frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
