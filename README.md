# Train Age Gender Model
## Train, valid, test split
We use 80% data for training and 20% for testing.
## Preprocessing
We transform data through the pipeline of transformation as we mention in the report (Section 3.1) 
and build train datasets and test datasets.
## Training stage:
1. We build the family of the ResNet and VGG model: ResNet-18,-34,-50,-101,-152 and VGG -16, -19.
2. Training configuration:\
• Size of input image : 200x200x3\
• Number channel of input image: 3\
• Number outputs: 2 labels for Gender and an integer for Age\
• Learning rate: 0.0003\
• Batch size: 32\
• Number of Epochs: 150	\
• Optim	izer: Adam\
3. We will save model parameters checkpoint each times we have the better validation loss.
## Evaluation:
The model's performance is assessed using the following metrics:

* Age Prediction:
MAE (Mean Absolute Error): Measure of the average difference between predicted and actual ages.

* Gender Prediction:
Binary Crossentropy: Measures the performance of a classification model, specifically for binary classification like gender prediction.
## Infer:
After training the model, we can make prediction for an image by forwarding the image into model and obtain the results

1. Importing
    ```
    from FaceRecog.predict import predict as age_gender_predict
    import tensoflow as tf
    ```
2. Load model
    ```
    agegenderModel = tf.keras.models.load_model(r"FaceRecog/pretrained/agegender34.h5")
    ```
3. Prediction
    ```
    pred_gender, pred_age, pred_age_cat = age_gender_predict(faceImg, agegenderModel)
    ```


# Train Facial Emote Recognition Model
## Train, valid, test split
We use 80% data for training, 10% for validation and 10% for testing.
## Preprocessing
We transform data through the pipeline of transformation as we mention in the report (Section 3.2) 
and build the dataloader for train datasets, valid datasets and test datasets.
## Training stage:
   1. We build the family of the ResNet and VGG model: ResNet-18,-34,-50,-101,-152 and VGG -16, -19.
   2. Training configuration:\
	• Size of input image : 40x40x1\
	• Number channel of input image: 1\
	• Number outputs: 7\
	• Learning rate: 0.001\
	• Batch size: 64 (We crop one 48x48 pixels image into 5 crops 40x40 image. 
			  Therefore, we have batch size 64*5 = 320 of 40x40 image)\
	• Number of Epochs: 150	\
	• Optim	izer: Adam\
	• Weight decay : 1e-4\
	• Scheduler : ReduceLROnPlateau. We wil reduce the learning rate by 0.5 if there are no improvement
	of the accuracy in validation datasets after 5 epochs.\
    3. We will save model parameters checkpoint each times we have the better valid accuracy.
## Evaluation:

1. We compute the train loss and train accuracy for each epochs which equals the averge the train loss and train accuracy of each batch in that epochs.
2. We also compute the valid loss and valid accuracy in the valid datasets after each epoch. ( Use valid accuracy help scheduler decide).
3. After the training phase, we compute the test loss and test accuracy in the test datasets. Then we use it to evaluate the performance of each model.

## Infer:
After training the model, we can make prediction for an image by forwarding the image into model and obtain the results

1. Importing
    ```
    from FaceEmote.predict import predict as emote_predict, load_model
    ```
2. Load model
    ```expression_model = ResNet18()
    optimizer = torch.optim.SGD(expression_model.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-4, nesterov=True)

    expression_model, optimizer = load_model(expression_model, optimizer, "FaceEmote/pretrained/Express_model_final.pth")
    ```
3. Prediction
    ```
    emote_prediction = emote_predict(faceLoad[i], expression_model)
    ```

# Train YOLO
## Train, valid, test split
We use 80% data for training, 10% for validation and 10% for testing.
## Preprocessing
Ultralytics' YOLOv5 model is used. As a result, we must preprocess our data into a suitable format.\
We must establish folders called "images" and "labels" to hold images and labels.\
Each folder contains graphics and labels for the train, valid, and test conditions.\
Label files are in txt format and have the following syntax for each bounding box: "class x_center y_center width height".\
There is only one class in our problem, which is "face." As a result, class=0 for all bounding boxes.\
Because of the YOLOv5 requirement, we resize images to 640x640.
## Training and Evaluating
We create a config.yaml file to specify our configuration. The content of config.yaml:\
\# Train/val/test sets \
train: '/kaggle/working/images/train'\
val: '/kaggle/working/images/valid'\
test: '/kaggle/working/images/test'\
\
\# Classes\
names:\
  0: face

Next, we need to import YOLO from ultralytics
```
from ultralytics import YOLO
```
If you have not installed ultralytics, you need to install it first
```
pip install ultralytics
```
We defined our model
```
model=YOLO('yolov5n.yaml')
```
Notice that in this project, we use nano version of YOLOv5. You can use small version(yolov5s.yaml), medium(yolov5m.yaml), large(yolov5l.yaml),  or extra large(yolov5x.yaml)
Then train model by
```
results=model.train(data=config_path, epochs=100, resume=True, iou=0.5, conf=0.001,save=True)
```
## Infer
After training the model, we can make prediction for an image by forwarding the image into model and extract the bounding boxes from the results
```
results = model(image_path)
for result in results:
    res_plotted = result.plot()
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
```
You can load our trained model save in "best.pt"
```
model=YOLO('best.pt')
```
## Evaluation
Ultralytics evaluate automatically the performance of the model. All you need to do is logging in to your wandb account and visualize the loss, precision, recall, mean average precision.


# Checkpoint
All checkpoints are saved at https://drive.google.com/drive/u/0/folders/1OIZZL7P-rIghrqzWOKJDQeqUrX_g9HHu