import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

frame = cv2.imread('carpark.png')
frame = cv2.resize(frame, (1020, 500))

my_file = open("clz.txt", "r")
data = my_file.read()
class_list = data.split("\n")
   

area1=[(194, 412),(117, 322),(313, 301),(453, 400)]

area2=[(495, 207),(403, 171),(549, 153),(630, 195)]

area3=[(700, 373),(520, 281),(689, 267),(894, 346)]



results = model.predict(frame)

# Process the predictions
a = results[0].boxes.data
px = pd.DataFrame(a).astype("float")

# Prepare lists for each parking slot
list1, list2, list3, list4, list5, list6, list7, list8, list9, list10, list11, list12 = ([] for _ in range(12))

for index, row in px.iterrows():
    x1, y1, x2, y2, _, d = row
    c = class_list[int(d)]
    
    if 'car' in c:
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2

        # Check which parking area the car is in and update the respective lists
        if cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list1.append(c)
            cv2.putText(frame, str(c), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        if cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list2.append(c)

        if cv2.pointPolygonTest(np.array(area3, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list3.append(c)

        

# Calculate occupancy
a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 = len(list1), len(list2), len(list3), len(list4), len(list5), len(list6), len(list7), len(list8), len(list9), len(list10), len(list11), len(list12)

if a1==1:
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('1'),(50,441),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('1'),(50,441),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
if a2==1:
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('2'),(106,440),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('2'),(106,440),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
if a3==1:
    cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('3'),(175,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('3'),(175,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)


# Show the image with predictions
cv2.imshow("Parking Slots", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the status of each parking slot
print(f"Parking slot 1: {a1} cars")
print(f"Parking slot 2: {a2} cars")
print(f"Parking slot 3: {a3} cars")
print(f"Parking slot 4: {a4} cars")
print(f"Parking slot 5: {a5} cars")
print(f"Parking slot 6: {a6} cars")
print(f"Parking slot 7: {a7} cars")
print(f"Parking slot 8: {a8} cars")
print(f"Parking slot 9: {a9} cars")
print(f"Parking slot 10: {a10} cars")
print(f"Parking slot 11: {a11} cars")
print(f"Parking slot 12: {a12} cars")