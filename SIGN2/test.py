import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the region of interest from the original frame
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        try:
            # Check for 'A' sign (straight closed fist)
            fingers = detector.fingersUp(hand)
            if fingers == [0, 0, 0, 0, 0]:  # All fingers down (fist closed)
                label = "A"
            elif fingers == [1, 0, 0, 0, 0]:  # Only thumb up
                label = "Help"
            elif fingers == [1, 0, 0, 0, 1]:  # Thumb and little finger up
                label = "Y"
            else:
                # Detect other letters based on aspect ratio
                aspectRatio = h / w
                if aspectRatio > 1.8:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:imgSize, wGap:wCal + wGap] = imgResize
                    label = "B"
                elif 0.5 <= aspectRatio < 1.0:
                    imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
                    imgWhite = imgResize
                    label = "C"
                else:
                    label = None

            if label:
                cv2.putText(imgOutput, label, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)

        except Exception as e:
            print("Error processing image:", e)

    else:
        label = None

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

