import cv2
import os

# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="t9oH2x69y79IXjOWbxTF"
)

names=os.listdir("C:\\Users\\sanse\\Desktop\\number")

for name in names:
    print(name)
    # image_path="orig3.jpg"
    image_path="C:\\Users\\sanse\\Desktop\\number\\"+name

    # infer on a local image
    result = CLIENT.infer(image_path, model_id="digits-detection-yuayi/3")
    sorted_predictions = sorted(result['predictions'], key=lambda pred: pred['x'], reverse=False)
    img=cv2.imread(image_path)

    # img = cv2.resize(img,(175,91), interpolation=cv2.INTER_AREA)

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
    counter=0
    for crop in cropped:
        # cv2.imshow("crop",crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(f"digits/{name[:-4]}{counter}.jpg",crop)
        counter+=1