from inference_sdk import InferenceHTTPClient
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import os
import time

CLIENT_DETECT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="t9oH2x69y79IXjOWbxTF"
)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path,detect_confidence):
    CLIENT_DETECT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="t9oH2x69y79IXjOWbxTF"
    )
    result = CLIENT_DETECT.infer(image_path, model_id="digits-detection-yuayi/3")
    sorted_predictions = sorted(result['predictions'], key=lambda pred: pred['x'], reverse=False)
    img = cv2.imread(image_path)

    cropped = []
    text_pos = []

    for pred in sorted_predictions:
        cv2.rectangle(img, (int(pred['x']) - int(pred['width']) // 2, int(pred['y']) - int(pred['height']) // 2),
                      (int(pred['x']) + int(pred['width']) // 2, int(pred['y']) + int(pred['height']) // 2),
                      (255, 0, 0), 1)
        text_pos.append((int(pred['x']) - int(pred['width']) // 2, int(pred['y']) - int(pred['height']) // 2))
        if float(pred['confidence']) > detect_confidence:
            cropped.append(img[int(pred['y']) - int(pred['height']) // 2:int(pred['y']) + int(pred['height']) // 2,
                           int(pred['x']) - int(pred['width']) // 2:int(pred['x']) + int(pred['width']) // 2])
    counter = 0
    for crop in cropped:
        cv2.imwrite(f".temp/{counter}.jpg", crop)
        counter += 1

    number = ""
    names = os.listdir(".temp")

    model = load_model('digits_shit_hands.keras')
    for name in names:
        img_path = f'.temp/{name}'
        prepared_image = load_and_preprocess_image(img_path)
        predictions = model.predict(prepared_image)
        predicted_class = np.argmax(predictions, axis=1)
        number += str(predicted_class[0])

    for name in names:
        os.remove(f".temp/{name}")
    return number

counter=0
start_time = time.time()
with open("test/labels.txt","r") as file:
    for line in file:
        name=line.split("-")[0]
        label=line.split("-")[1]

        res =predict(f"test/{name}.jpg",0.50)

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