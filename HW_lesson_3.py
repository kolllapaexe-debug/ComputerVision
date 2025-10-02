import cv2
img = cv2.imread("images/face.jpg")
rez = cv2.resize(img, (500, 600))
cv2.rectangle(rez, (150, 197), (344, 446), (0, 0, 255), 2)
cv2.putText(rez, "Baranov Yehor", (190, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
cv2.imshow("Image", rez)
cv2.waitKey(0)
cv2.destroyAllWindows()