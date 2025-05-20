import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import os
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
folder = "DATAIMG/A"
counter = 0

# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Find hands in the frame
    hands, imgHands = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the region of interest from the original frame
        imgCrop = imgHands[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize and place the cropped image onto the white background
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:imgSize, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :imgSize] = imgResize

        # Display the cropped image and the white background
        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        filename = f'{folder}/Image_{int(time.time() * 1000)}.jpg'  # Unique filename
        cv2.imwrite(filename, imgWhite)
        print(f"Image saved: {filename}")
        print(f"Counter: {counter}")

    elif key == ord("q"):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()