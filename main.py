import os

import cv2
import ultralytics
from dotenv import load_dotenv

# в файле .env хранится путь к видеофайлу
load_dotenv()
video_path = os.getenv("VIDEO_PATH")

# взял medium модель
model = ultralytics.YOLO("yolo11m.pt")

# классы: машина, мотоцикл, грузовик
classes = [2, 3, 7]

capture = cv2.VideoCapture(video_path)

mask = None

while True:
    ret, frame = capture.read()

    if not ret:
        break

    # формируем маску только в первом проходе
    if mask is None:
        # добавляем маску, чтобы модель детектила местность, которая ближе к шлагбаумам
        mask = cv2.imread("masks/binary_mask.png", cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    result = model(frame, classes=classes)
    result = result[0]

    vehicles_detected = []

    for box in result.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0]
        # центр бокса
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        # проверяем, что центр попал в белую область маски
        if mask[cy, cx]:
            vehicles_detected.append(box)

    annotated_frame = result.plot()

    annotated_frame = cv2.bitwise_and(
        annotated_frame, annotated_frame, mask=mask
    ) + cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    found = bool(vehicles_detected)

    # добавляем текст на изображение о количестве ТС
    text = f"{len(vehicles_detected)} vehicles detected" if found else "No vehicles detected"

    cv2.putText(
        annotated_frame,
        text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0) if found else (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Video", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break