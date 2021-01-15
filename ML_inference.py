import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

#Function to detect faces
def detect_masks(frame, faceNet, maskNet):

    # getting dimensions of the frame and generating a blob
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))


    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    face_locations = []
    face_predictions = []

    # looping over frames with detected faces
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(width - 1, endX), min(height - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            face_locations.append((startX, startY, endX, endY))

            if len(faces) > 0:
                face_predictions = maskNet.predict(faces)

            return (face_locations, face_predictions)

# Starts video stream from user's webcam and sends frames to detect_masks function

def start_video_stream(faceNet, maskNet):
    stream = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        frame = stream.read()
        frame = imutils.resize(frame, width=400)

        (locs, preds) = detect_masks(frame, faceNet, maskNet)

        for (box, pred) in zip(locs,preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "With Face Mask" if mask > withoutMask else "Without Face Mask"
            color = (0, 255, 0) if label == "With Face Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    vs.stop()

# load models for face and mask detectors
# face detector model
face_prototxt_path = "./face_detector/deploy.prototxt"
face_weights_path = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(face_prototxt_path, face_weights_path)
# mask detector model
# "./pretrained_mask_detector.model"
mask_model_path = "./mask_detector.model"
maskNet = load_model(mask_model_path)

#Run video stream and face detection function
start_video_stream(faceNet, maskNet)
