import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)
#rgb = bgr
#1st way
# img[:] = 135, 62, 58
#2nd way
# img[100:150, 200:280] = 135, 62, 58
#3rd way
cv2.rectangle(img,(100,100),(200,200),(135, 62, 58), 2)

cv2.line(img,(100,100),(200,200),(135, 62, 58),2)
print(img.shape)
cv2.line(img, (0, img.shape[0]//2), (img.shape[1], img.shape[0]//2), (135, 62, 58), 2)
cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (135, 62, 58), 2)
cv2.circle(img, (200,200), 20, (135, 62, 58), 2)
cv2.putText(img, "babanoushka", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (135, 62, 58))

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()