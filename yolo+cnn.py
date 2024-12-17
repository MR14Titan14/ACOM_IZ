from inference_sdk import InferenceHTTPClient
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import os
import time
from ultralytics import YOLO

def predict(image_path):
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

    model = load_model('digits_shit_best.keras')
    for name in names:
        img_path = f'.temp/{name}'
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
        image = image.reshape(1, 28, 28, 1)
        image = image / 255.0


        # img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        # img_array = image.img_to_array(img)
        # img_array = img_array / 255.0
        # img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)
        number += str(predicted_class[0])

    for name in names:
        os.remove(f".temp/{name}")

    if len(number)>6:
        num = number[:6]
    else:
        num = number
    return num

counter=0
start_time = time.time()
with open("test/labels.txt","r") as file:
    for line in file:
        name=line.split("-")[0]
        label=line.split("-")[1]

        res =predict(f"test/{name}.jpg")

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