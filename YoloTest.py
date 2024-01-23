from ultralytics import YOLO
import cv2
import cvzone

capture = cv2.VideoCapture(0) # capturing webcam
# capture = cv2.VideoCapture("../Videos/bikes.mp4")
width = 1280
height = 720
capture.set(3, width)  # property id for width is 3
capture.set(4, height)  # height is 4

model = YOLO('../MyWeights/yolov8n.pt')


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = capture.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (233, 54, 23), 3)
            # confidence rectangles
            confidence = round(float(box.conf[0]), 2)
            print(confidence)
            # class name
            cls = box.cls[0]
            print(classNames[int(cls)])
            # cvzone.putTextRect(img, f"{classNames[int(cls)]}", (max(x1, 0), min(y2, height)))
            cvzone.putTextRect(img, f"{classNames[int(cls)]},{confidence}",
                               (max(x1, 0), max(y1 - 20, 30)))  # don't have to use cvzone, but looks nicer


    cv2.imshow("Image", img)
    cv2.waitKey(1)