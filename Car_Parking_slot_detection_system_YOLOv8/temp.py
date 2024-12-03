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

frame = cv2.imread('img.png')
frame = cv2.resize(frame, (1020, 500))

my_file = open("clz.txt", "r")
data = my_file.read()
class_list = data.split("\n")
   

area1=[(52,364),(30,417),(73,412),(88,369)]

area2=[(105,353),(86,428),(137,427),(146,358)]

area3=[(159,354),(150,427),(204,425),(203,353)]

area4=[(217,352),(219,422),(273,418),(261,347)]

area5=[(274,345),(286,417),(338,415),(321,345)]

area6=[(336,343),(357,410),(409,408),(382,340)]

area7=[(396,338),(426,404),(479,399),(439,334)]

area8=[(458,333),(494,397),(543,390),(495,330)]

area9=[(511,327),(557,388),(603,383),(549,324)]

area10=[(564,323),(615,381),(654,372),(596,315)]

area11=[(616,316),(666,369),(703,363),(642,312)]

area12=[(674,311),(730,360),(764,355),(707,308)]


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

        if cv2.pointPolygonTest(np.array(area4, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list4.append(c)

        if cv2.pointPolygonTest(np.array(area5, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list5.append(c)

        if cv2.pointPolygonTest(np.array(area6, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list6.append(c)

        if cv2.pointPolygonTest(np.array(area7, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list7.append(c)

        if cv2.pointPolygonTest(np.array(area8, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list8.append(c)

        if cv2.pointPolygonTest(np.array(area9, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list9.append(c)

        if cv2.pointPolygonTest(np.array(area10, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list10.append(c)

        if cv2.pointPolygonTest(np.array(area11, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list11.append(c)

        if cv2.pointPolygonTest(np.array(area12, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            list12.append(c)

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
if a4==1:
    cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('4'),(250,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('4'),(250,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
if a5==1:
    cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('5'),(315,429),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('5'),(315,429),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
if a6==1:
    cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('6'),(386,421),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('6'),(386,421),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1) 
if a7==1:
    cv2.polylines(frame,[np.array(area7,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('7'),(456,414),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area7,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('7'),(456,414),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
if a8==1:
    cv2.polylines(frame,[np.array(area8,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('8'),(527,406),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area8,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('8'),(527,406),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)  
if a9==1:
    cv2.polylines(frame,[np.array(area9,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('9'),(591,398),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area9,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('9'),(591,398),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
if a10==1:
    cv2.polylines(frame,[np.array(area10,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('10'),(649,384),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area10,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('10'),(649,384),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
if a11==1:
    cv2.polylines(frame,[np.array(area11,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('11'),(697,377),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area11,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('11'),(697,377),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
if a12==1:
    cv2.polylines(frame,[np.array(area12,np.int32)],True,(0,0,255),2)
    cv2.putText(frame,str('12'),(752,371),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
else:
    cv2.polylines(frame,[np.array(area12,np.int32)],True,(0,255,0),2)
    cv2.putText(frame,str('12'),(752,371),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)


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