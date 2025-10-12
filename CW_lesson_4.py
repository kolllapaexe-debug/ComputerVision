import cv2
import numpy as np

from praktichna import text_y

img = cv2.imread("images/123.jpg")
scale = 1
img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale))
print(img.shape)
img_copy_collor = img.copy()
img_copy=img.copy()

img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 4)
img_copy = cv2.equalizeHist(img_copy)
img_copy = cv2.Canny(img_copy, 100, 150)
cont, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # зовнішні контури - кіцеві точки контурів
#maluvana konturiv pramk ta textu
for cnt in cont:
    area = cv2.contourArea(cnt)
    if area > 65:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img_copy_collor, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img_copy_collor, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y-5 > 10 else y + 15
        text = f'x:{x}  y:{y}  S:{int(area)}'
        cv2.putText(img_copy_collor, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)




#cv2.imshow("image copy", img_copy)
#cv2.imshow("image", img)
cv2.imshow("image copy collor", img_copy_collor)
cv2.waitKey(0)
cv2.destroyAllWindows()
