import cv2
import imutils
import numpy as np
import pytesseract

widthImg = 600
heightImg = 400
pytesseract.pytesseract.tesseract_cmd = r'c:\\Program Files\\Tesseract-OCR\\tesseract.exe'
dir = r'c:\Users\hp\PycharmProjects\plate\dataset'
photo_path = 'p/photo_2021-03-25_16-19-05 (2).jpg'
image = cv2.imread(photo_path, cv2.IMREAD_COLOR)
image = cv2.resize(image, (widthImg, heightImg))

img = cv2.imread(photo_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (widthImg, heightImg))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15)

edged = cv2.Canny(gray, 30, 200)
kernel = np.ones((5, 5))
imgDial = cv2.dilate(edged, kernel, iterations=2)
imgThres = cv2.erode(imgDial, kernel, iterations=1)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
new_image = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
# print((topx,topy), (bottomx, bottomy))
Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


biggest = reorder(screenCnt)
# print(biggest)
pts1 = np.float32(biggest)
pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(image, matrix, (widthImg, heightImg))
b = biggest

imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
imgCropped = cv2.resize(imgCropped, ((abs(10+b[1][0][0]-b[0][0][0]), abs(b[0][0][1]-b[2][0][1]))))

# (abs(b[1][0][0]-b[0][0][0]), abs(b[0][0][1]-b[2][0][1]))

text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("License Plate Recognition\n")
print("Detected number plate Number is:", text)
img = cv2.resize(img, (500, 300))
Cropped = cv2.resize(Cropped, (400, 200))
# cv2.imwrite(filename=rf'{dir}\text.png ', img=Cropped)
# crop = getWarp(img, screenCnt)
cv2.imshow('Car', img)
cv2.imshow('Cropped', Cropped)
# cv2.imshow('output', imgOutput)
# cv2.imshow('gray', gray)
cv2.imshow('crrop', imgCropped)
cv2.waitKey(0)
