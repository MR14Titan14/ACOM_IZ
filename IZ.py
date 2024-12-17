import cv2
import os

from inference_sdk import InferenceHTTPClient

def predict_roboflow(image_path,detect_confidence,rec_confidence):
    CLIENT_DETECT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="t9oH2x69y79IXjOWbxTF"
    )

    CLIENT_REC = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="w1MC4PGBcnlYHRGpuk4g"
    )


    result = CLIENT_DETECT.infer(image_path, model_id="digits-detection-yuayi/3")
    sorted_predictions = sorted(result['predictions'], key=lambda pred: pred['x'], reverse=False)
    img=cv2.imread(image_path)

    cropped=[]
    text_pos=[]

    for pred in sorted_predictions:
        cv2.rectangle(img,(int(pred['x'])-int(pred['width'])//2,int(pred['y'])-int(pred['height'])//2),(int(pred['x'])+int(pred['width'])//2,int(pred['y'])+int(pred['height'])//2),(255,0,0),1)
        text_pos.append((int(pred['x'])-int(pred['width'])//2,int(pred['y'])-int(pred['height'])//2))
        if float(pred['confidence'])>detect_confidence:
            cropped.append(img[int(pred['y'])-int(pred['height'])//2:int(pred['y'])+int(pred['height'])//2,int(pred['x'])-int(pred['width'])//2:int(pred['x'])+int(pred['width'])//2])
    counter=0
    for crop in cropped:
        cv2.imwrite(f".temp/{counter}.jpg",crop)
        counter+=1

    number=""
    names=os.listdir(".temp")
    for name in names:
        result = CLIENT_REC.infer(f".temp/{name}", model_id="digits_shit/1")
        if float(result['predictions'][0]['confidence'])>rec_confidence:
            number+=result['predictions'][0]['class']

    for name in names:
        os.remove(f".temp/{name}")
    # print(number)

    for i in range(len(text_pos)):
        cv2.putText(img,number[i],text_pos[i],cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0),1)

    resized=cv2.resize(img,(320,240),interpolation=cv2.INTER_AREA)

    # cv2.imshow("win",resized)
    cv2.imwrite(".temp/res.jpg",resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return number, resized
num,img=predict_roboflow("orig.jpg", 0.40, 0.75)
print(num)