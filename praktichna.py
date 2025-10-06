import cv2
import numpy as np

card_width, card_height = 600, 400
background_color_light_blue = (230, 200, 150)
bezel_color_gray = (54, 31, 21)
text_color = (40, 40, 40)
white_color = (255, 255, 255)

image = np.full((card_height, card_width, 3), background_color_light_blue, dtype=np.uint8)

frame_thickness = 5
cv2.rectangle(image, (0, 0), (card_width - 1, card_height - 1), bezel_color_gray, frame_thickness)

profile_pic_path = "images/face.jpg"
profile_pic = cv2.imread(profile_pic_path)

profile_pic_resized = cv2.resize(profile_pic, (120, 120))
image[30:150, 30:150] = profile_pic_resized

name_text = "Baranov Yehor"
cv2.putText(image, name_text, (180, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)

position_text = "SOFTWARE ENGINEER"
cv2.putText(image, position_text, (180, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

email_text = "Email: kolllapa.exe@gmail.com"
phone_text = "Phone: +380 50 263 48 03"
dob_text = "Date of Birth: 12/11/2009"

cv2.putText(image, email_text, (180, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
cv2.putText(image, phone_text, (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
cv2.putText(image, dob_text, (180, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

qr_code_path = "images/qrcode.png"
qr_img_cv = cv2.imread(qr_code_path)

qr_img_cv = cv2.resize(qr_img_cv, (120, 120))

qr_x_pos = card_width - qr_img_cv.shape[1] - 25
qr_y_pos = 230
image[qr_y_pos: qr_y_pos + qr_img_cv.shape[0], qr_x_pos: qr_x_pos + qr_img_cv.shape[1]] = qr_img_cv

bottom_bar_height = 40
cv2.rectangle(image, (0, card_height - bottom_bar_height), (card_width, card_height), bezel_color_gray, -1)

bottom_text = "OPENCV BUSINESS CARD"
(text_width, text_height), baseline = cv2.getTextSize(bottom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
text_x = (card_width - text_width) // 2
text_y = card_height - bottom_bar_height + (bottom_bar_height + text_height) // 2 - baseline + 8

cv2.putText(image, bottom_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white_color, 2, cv2.LINE_AA)

cv2.imwrite("images/business_card.png", image)

cv2.imshow("Business Card", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

