from inference_sdk import InferenceHTTPClient
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import os
import time
from ultralytics import YOLO

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path, pr):
    model = YOLO('model.pt')
    img = cv2.imread(image_path)
    results = model(img)
    cropped = []

    # Отображение результатов
    for result in results:
        boxes = result.boxes.xyxy  # Координаты боксов
        sorted_boxes = sorted(boxes, key=lambda box: box[0])
        for box in sorted_boxes:
            x1, y1, x2, y2 = box  # Распаковка координат и информации о классе
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Рисуем рамку
            cropped.append(img[int(y1):int(y2),int(x1):int(x2)])


    counter = 0
    for crop in cropped:
        cv2.imwrite(f".temp/{counter}.jpg", crop)
        counter += 1

    number = ""
    names = os.listdir(".temp")

    CLIENT_REC = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="w1MC4PGBcnlYHRGpuk4g"
    )

    for name in names:
        result = CLIENT_REC.infer(f".temp/{name}", model_id="digits_shit/1")
        if float(result['predictions'][0]['confidence']) > pr:
            number += result['predictions'][0]['class']
        else:
            number += ""

    for name in names:
        os.remove(f".temp/{name}")
    return number

counter=0
start_time = time.time()
with open("test/labels.txt","r") as file:
    for line in file:
        name=line.split("-")[0]
        label=line.split("-")[1]

        res =predict(f"test/{name}.jpg", pr=0.90)

        label=label.replace("\n","")
        print(f"label: {label}")
        print(f"res: {res}")
        if(res==label):
            print("______")
            counter+=1
end_time = time.time()
interval = end_time - start_time
print(f"{counter}%")
print(f"{interval}")