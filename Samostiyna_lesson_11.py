import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 128, 255),
    "purple": (255, 0, 255),
    "pink": (180, 105, 255),
    "white": (255, 255, 255)
}


x = []
y = []
noise_range = 20
samples_per_color = 50

for color_name, bgr_values in colors.items():
    for _ in range(samples_per_color):
        noise = np.random.randint(-noise_range, noise_range + 1, 3)
        noise_bgr = np.array(bgr_values) + noise
        noise_bgr = np.clip(noise_bgr, 0, 255)
        x.append(noise_bgr)
        y.append(color_name)

x = np.array(x)
y = np.array(y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)


def get_shape(corners, aspect_ratio):
    shape = "null"

    if corners == 3:
        shape = "triangle"
    elif corners == 4:
        if 0.95 <= aspect_ratio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"
    elif corners > 5:
        shape = "circle"

    return shape


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    result_frame = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bright = np.array([0, 80, 80])
    upper_bright = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_bright, upper_bright)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_counts = {}

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        corners = len(approx)
        aspect_ratio = float(w) / h
        shape = get_shape(corners, aspect_ratio)
        roi_frame = frame[y:y + h, x:x + w]
        roi_mask = mask[y:y + h, x:x + w]
        mean_color_bgr = cv2.mean(roi_frame, mask=roi_mask)[:3]

        color_label = model.predict([mean_color_bgr])[0]
        color_counts[color_label] = color_counts.get(color_label, 0) + 1

        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = f"{color_label} {shape}"
        cv2.putText(result_frame, text, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)

    sum_items = [f"{count} {color}" for color, count in color_counts.items()]
    sum_text = "Summary: " + ", ".join(sum_items)
    cv2.putText(result_frame, sum_text, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1)
    cv2.imshow("result", result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()