import cv2
import os

# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="t9oH2x69y79IXjOWbxTF"
)

# image_path="orig3.jpg"
image_path="orig.jpg"

# infer on a local image
result = CLIENT.infer(image_path, model_id="digits-detection-yuayi/3")
sorted_predictions = sorted(result['predictions'], key=lambda pred: pred['x'], reverse=False)
img=cv2.imread(image_path)

print(result)

cropped=[]

for pred in sorted_predictions:
    # print(pred['x'],pred['y'],pred['width'],pred['height'])
    # cv2.circle(img,(int(pred['x']),int(pred['y'])),3,(0,255,0),1)
    # cv2.rectangle(img,(int(pred['x'])-int(pred['width'])//2,int(pred['y'])-int(pred['height'])//2),(int(pred['x'])+int(pred['width'])//2,int(pred['y'])+int(pred['height'])//2),(255,0,0),2)
    if float(pred['confidence']>0.65):
        cropped.append(img[int(pred['y'])-int(pred['height'])//2:int(pred['y'])+int(pred['height'])//2,int(pred['x'])-int(pred['width'])//2:int(pred['x'])+int(pred['width'])//2])
# cv2.imshow("win",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()