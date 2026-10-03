import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import easyocr
import os
from datetime import datetime
import mysql.connector
import re
import tempfile

# ---------------- INIT ----------------
reader = easyocr.Reader(['en'])

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='detect'
)
cursor = conn.cursor()

# MODELS
model_main = YOLO("yolov8l.pt")  # person + bike
model_helmet = YOLO("../weights/best.pt")
model_plate = YOLO("../weights/number_plate_model.pt")
model_seatbelt = YOLO("../weights/seatbelt.pt")

save_path = "violations"
os.makedirs(save_path, exist_ok=True)

st.title(" Smart Traffic Violation System")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg","jpeg","png","mp4","avi","mov"]
)

# ---------------- UTILS ----------------
def is_near(box1, box2, margin=50):
    x1,y1,x2,y2 = box1
    a1,b1,a2,b2 = box2
    return (a1 < x2+margin and a2 > x1-margin and
            b1 < y2+margin and b2 > y1-margin)

# ---------------- CORE ----------------
def process_frame(frame, challan_cache):

    results = model_main(frame,conf=0.25)
    results_helmet = model_helmet(frame,conf=0.25)
    results_plate = model_plate(frame,conf=0.25)
    results_seatbelt = model_seatbelt(frame,conf=0.25)

    person_boxes, bike_boxes, car_boxes = [], [], []
    helmet_boxes = []
    seatbelt_boxes = []

    # -------- MAIN DETECTION --------
    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            x1,y1,x2,y2 = map(int, b.xyxy[0])

            if cls == 0:  # person
                person_boxes.append((x1,y1,x2,y2))
            elif cls == 3:  # bike
                bike_boxes.append((x1,y1,x2,y2))
            elif cls == 2:  # car
                car_boxes.append((x1,y1,x2,y2))

    # -------- HELMET --------
    for r in results_helmet:
        for b in r.boxes:
            cls = int(b.cls[0])
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            if cls == 0:
                helmet_boxes.append((x1,y1,x2,y2))

    # -------- SEATBELT --------
    for r in results_seatbelt:
        for b in r.boxes:
            cls = int(b.cls[0])
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            label = model_seatbelt.names[cls]
            seatbelt_boxes.append((label,(x1,y1,x2,y2)))

    # -------- NUMBER PLATE --------
    plate_no = "UNKNOWN"
    for r in results_plate:
        for b in r.boxes.xyxy:
            x1,y1,x2,y2 = map(int,b)
            crop = frame[y1:y2,x1:x2]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
            cv2.putText(frame,"PLATE",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
            text = reader.readtext(crop, detail=0)
            temp = re.sub(r'[^A-Z0-9]', '', "".join(text).upper())
            if temp:
                plate_no = temp
                break
            print("Plate detections:", results_plate[0].boxes)

    # -------- PERSON LOGIC --------
    for person in person_boxes:
        px1,py1,px2,py2 = person

        on_bike = any(is_near(person, bike) for bike in bike_boxes)
        in_car = any(is_near(person, car) for car in car_boxes)

        # -------- BIKE LOGIC --------
        if on_bike:
            has_helmet = any(is_near(person,h) for h in helmet_boxes)

            if has_helmet:
                color = (0,255,0)
                label = "Helmet"
            else:
                color = (0,0,255)
                label = "No Helmet"

                # CHALLAN
                key = (px1//50, py1//50, "No Helmet")
                if key not in challan_cache:
                    challan_cache.add(key)

                    filename = f"helmet_{datetime.now().strftime('%H%M%S')}.jpg"
                    filepath = os.path.join(save_path, filename)
                    cv2.imwrite(filepath, frame)

                    cursor.execute(
                        "INSERT INTO challans (plate, violation_type, fine, image_path) VALUES (%s,%s,%s,%s)",
                        (plate_no, "No Helmet", 1000, filepath)
                    )
                    conn.commit()

                    st.error(f" No Helmet | Plate: {plate_no}")

        # -------- CAR LOGIC --------
        elif in_car:
            has_seatbelt = None

            for label_sb, box in seatbelt_boxes:
                if is_near(person, box):
                    if "no" in label_sb.lower():
                        has_seatbelt = False
                    else:
                        has_seatbelt = True

            if has_seatbelt:
                color = (0,255,0)
                label = "Seatbelt OK"
            else:
                color = (0,165,255)
                label = "No Seatbelt"
           

                # CHALLAN
                key = (px1//50, py1//50, "No Seatbelt")
                if key not in challan_cache:
                    challan_cache.add(key)

                    filename = f"seatbelt_{datetime.now().strftime('%H%M%S')}.jpg"
                    filepath = os.path.join(save_path, filename)
                    cv2.imwrite(filepath, frame)

                    cursor.execute(
                        "INSERT INTO challans (plate, violation_type, fine, image_path) VALUES (%s,%s,%s,%s)",
                        (plate_no, "No Seatbelt", 1000, filepath)
                    )
                    conn.commit()

                    st.error(f" No Seatbelt | Plate: {plate_no}")

        # -------- UNKNOWN (IGNORE) --------
        else:
            continue

        # DRAW BOX
        cv2.rectangle(frame,(px1,py1),(px2,py2),color,2)
        cv2.putText(frame,label,(px1,py1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # -------- TRIPLING (ONLY BIKE) --------
    for bike in bike_boxes:
        count = sum(1 for p in person_boxes if is_near(bike,p))

        bx1,by1,bx2,by2 = bike
        cv2.rectangle(frame,(bx1,by1),(bx2,by2),(255,255,0),2)
        cv2.putText(frame,f"Persons: {count}",
                    (bx1,by1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

        if count >= 3:
            cv2.putText(frame,"TRIPLING!",(50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            st.warning(" Tripling Detected")

    return frame
# ---------------- MAIN ----------------
if uploaded_file:

    challan_cache = set()
    file_type = uploaded_file.type

    if "image" in file_type:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        output = process_frame(frame, challan_cache)
        st.image(output, channels="BGR")

    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame, challan_cache)
            stframe.image(frame, channels="BGR")

        cap.release()

