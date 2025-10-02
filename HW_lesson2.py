import cv2


img = cv2.imread("images/face.jpg")
email_img = cv2.imread("images/email.jpg")


resized_img = cv2.resize(img, (400, 400))
resized_email = cv2.resize(email_img, (400, 400))


gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
gray_email = cv2.cvtColor(resized_email, cv2.COLOR_BGR2GRAY)


edges_img = cv2.Canny(gray_img, 100, 200)
edges_email = cv2.Canny(gray_email, 100, 200)


cv2.imwrite("resized_photo.jpg", resized_img)
cv2.imwrite("grayscale_photo.jpg", gray_img)
cv2.imwrite("edges_photo.jpg", edges_img)

cv2.imwrite("resized_email.jpg", resized_email)
cv2.imwrite("grayscale_email.jpg", gray_email)
cv2.imwrite("edges_email.jpg", edges_email)
