from ultralytics import YOLO

model = YOLO('model.pt')

import cv2

image_path = 'C:\\Users\\sanse\\Desktop\\number\\1c6d16ab-1217.jpg'
image = cv2.imread(image_path)

image = cv2.resize(image,(640,640),interpolation=cv2.INTER_LINEAR)

results = model(image)

for result in results:
    boxes = result.boxes.xyxy
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

cv2.imwrite('output.jpg', image)
cv2.imshow('Detected Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()