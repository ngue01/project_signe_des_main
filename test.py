import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("hand_sign_detection_model_vgg16.h5", "labels_vgg16.txt")
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["Mal de tete", "Au Secours", "Vertige", "Vomissements"]

while True:
    success, img = cap.read()
    if not success:
        print("Erreur: Impossible de lire l'image de la caméra.")
        break
    
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            
            if 0 <= index < len(labels):
                print(prediction, index)
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
            else:
                print(f"Erreur: index de prédiction {index} hors des limites pour les labels.")
    
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
