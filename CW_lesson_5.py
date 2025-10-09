import cv2
import numpy as np
img = cv2.imread("images/kot.jpg")
img = cv2.resize(img, (480, 320))
img_copy = img.copy()
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([16, 0, 0]) #min hsv
upper = np.array([179, 255, 255]) #max hsv
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img, img, mask=mask)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w/ h, 2)
        #zaoktuglenist
        compactness = round((4*np.pi * area)/(perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.2*perimeter, True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Quadratic"
        elif len(approx) > 8:
            shape = "oval"
        else :
            shape = "nishe"
        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)
        cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f'shape:{shape}', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 0, 0), 2)
        cv2.putText(img_copy, f'Area{int(area)}, Perimeter:{int(perimeter)}', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 0, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 0, 0), 2)
cv2.imshow("img", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()