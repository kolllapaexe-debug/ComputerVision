import cv2
import numpy as np
img = cv2.imread("images/phorm.jpg")
img = cv2.resize(img, (680, 512))
img_copy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_probka = np.array([9, 0, 0])
upper_probka = np.array([179, 255, 255])
mask_probka = cv2.inRange(img, lower_probka, upper_probka)
lower_chudo = np.array([0, 0, 0])
upper_chudo = np.array([102, 255, 255])
mask_chudo = cv2.inRange(img, lower_chudo, upper_chudo)
lower_chprobka = np.array([0, 0, 0])
upper_chprobka = np.array([179, 255, 40])
mask_chprobka = cv2.inRange(img, lower_chprobka, upper_chprobka)
lower_car = np.array([38, 0, 0])
upper_car = np.array([68, 255, 182])
mask_car = cv2.inRange(img, lower_car, upper_car)
img = cv2.bitwise_and(img, img, mask=mask_car)
contours_probka, _ = cv2.findContours(mask_probka, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours_chprobka, _ = cv2.findContours(mask_chprobka, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours_chudo, _ = cv2.findContours(mask_chudo, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours_car, _ = cv2.findContours(mask_car, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for cnt in contours_probka:
    area = cv2.contourArea(cnt)
    if area > 100:
        perimeter = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if M["m00"] != 0:
            Cx = int(M["m10"] / M["m00"])
            Cy = int(M["m01"] / M["m00"])
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.2 * perimeter, True)

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)

for cnt in contours_chprobka:
    area = cv2.contourArea(cnt)
    if area > 5000:
        perimeter = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if M["m00"] != 0:
            Cx = int(M["m10"] / M["m00"])
            Cy = int(M["m01"] / M["m00"])
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.2 * perimeter, True)
        cv2.putText(img_copy, f'probka, Area{int(area)}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), )
        cv2.putText(img_copy, f'Area{int(area)}, Perimeter:{int(perimeter)}', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 0), )
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)

for cnt in contours_chudo:
    area = cv2.contourArea(cnt)
    if area > 130:
        perimeter = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if M["m00"] != 0:
            Cx = int(M["m10"] / M["m00"])
            Cy = int(M["m01"] / M["m00"])
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.2 * perimeter, True)

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)

for cnt in contours_car:
    area = cv2.contourArea(cnt)
    if area > 0:
        perimeter = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if M["m00"] != 0:
            Cx = int(M["m10"] / M["m00"])
            Cy = int(M["m01"] / M["m00"])
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.2 * perimeter, True)

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), 2)

cv2.imshow("img", img_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()


