import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
labels = ["B", "C"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    imgOutput = img.copy()
    # Find hands in the frame
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Calculate aspect ratio
        aspectRatio = h / w

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the region of interest from the original frame
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        try:
            # Resize and place the cropped image onto the white background
            if aspectRatio > 1.8:
                # If aspect ratio is high, classify as 'A'
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:imgSize, wGap:wCal + wGap] = imgResize
                label = "B"
            elif 1.0 <= aspectRatio <= 1.8:
                # If aspect ratio is medium, classify as 'B'
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                label = "A"
            elif 0.5 <= aspectRatio < 1.0:
                # If aspect ratio is low, classify as 'C'
                imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
                imgWhite = imgResize
                label = "C"
            else:
                label = None

            if label:
                # Display the predicted label on the output image
                cv2.putText(imgOutput, label, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # Display the cropped image and the white background (optional)
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)

        except Exception as e:
            print("Error processing image:", e)

    else:
        # If no hands are detected, do not display any label
        label = None

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()



