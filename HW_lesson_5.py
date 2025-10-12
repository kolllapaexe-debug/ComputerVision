import cv2
import numpy as np

def analyze_shape(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from path {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.erode(thresh, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = img.copy()
    shapes_data = []
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    placed_text_boxes = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        perimeter = cv2.arcLength(contour, True)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        x, y, w, h = cv2.boundingRect(contour)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        shape = "Unknown"
        compactness = (perimeter**2) / (4 * np.pi * area) if area != 0 else 0
        (cx_min_circle, cy_min_circle), radius_min_circle = cv2.minEnclosingCircle(contour)
        circle_area_ratio = area / (np.pi * (radius_min_circle ** 2)) if radius_min_circle > 0 else 0
        if circle_area_ratio > 0.85 and compactness < 1.15 and num_vertices > 8:
            shape = "Circle"
        elif num_vertices == 3:
            shape = "Triangle"
        elif num_vertices == 4:
            aspect_ratio_bbox = float(w) / h
            if 0.85 <= aspect_ratio_bbox <= 1.15:
                shape = "Square/Rhombus"
            else:
                shape = "Rectangle"
        elif num_vertices >= 5 and not cv2.isContourConvex(contour):
            shape = "Star"
        elif num_vertices > 4 and cv2.isContourConvex(contour):
            shape = "Polygon"
        else:
            shape = "Unknown"
        aspect_ratio = float(w) / h
        shapes_data.append({
            "Area": area,
            "Perimeter": perimeter,
            "Center": (cX, cY),
            "Shape": shape,
            "Aspect Ratio": aspect_ratio,
            "Compactness": compactness
        })
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 0, 0)
        bg_color = (255, 255, 255)
        line_height_base = int(font_scale * 20)
        line_spacing_extra = 5
        total_line_height = line_height_base + line_spacing_extra
        padding = 7
        text_lines = [
            f"Area: {area:.2f}",
            f"Perimeter: {perimeter:.2f}",
            f"Center: ({cX}, {cY})",
            f"Shape: {shape}",
            f"Aspect Ratio: {aspect_ratio:.2f}",
            f"Compactness: {compactness:.2f}"
        ]
        max_text_width = 0
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            if text_size[0] > max_text_width:
                max_text_width = text_size[0]
        text_block_height = len(text_lines) * total_line_height
        def check_overlap(new_box, existing_boxes):
            for existing_box in existing_boxes:
                if not (new_box[2] < existing_box[0] or
                        new_box[0] > existing_box[2] or
                        new_box[3] < existing_box[1] or
                        new_box[1] > existing_box[3]):
                    return True
            return False
        potential_positions = [
            (x, y - text_block_height - (padding * 2) - 10),
            (x, y + h + 10),
            (x + w + 10, y),
            (x - max_text_width - (padding * 2) - 10, y)
        ]
        best_pos = None
        for px, py in potential_positions:
            px = max(0, px)
            py = max(0, py)
            if px + max_text_width + (padding * 2) > output_img.shape[1]:
                px = output_img.shape[1] - (max_text_width + (padding * 2)) - 5
                px = max(0, px)
            if py + text_block_height + (padding * 2) > output_img.shape[0]:
                 py = output_img.shape[0] - (text_block_height + (padding * 2)) - 5
                 py = max(0, py)
            current_text_box = (px - padding, py - padding,
                                px + max_text_width + padding, py + text_block_height + padding)
            if not check_overlap(current_text_box, placed_text_boxes):
                best_pos = (px, py)
                break
        if best_pos is None:
            best_pos = (x, y - text_block_height - (padding * 2) - 10)
            best_pos = (max(0, best_pos[0]), max(0, best_pos[1]))
            if best_pos[0] + max_text_width + (padding * 2) > output_img.shape[1]:
                best_pos = (output_img.shape[1] - (max_text_width + (padding * 2)) - 5, best_pos[1])
            if best_pos[1] + text_block_height + (padding * 2) > output_img.shape[0]:
                best_pos = (best_pos[0], output_img.shape[0] - (text_block_height + (padding * 2)) - 5)
            best_pos = (max(0, best_pos[0]), max(0, best_pos[1]))
        text_start_x, text_start_y = best_pos
        final_text_box = (text_start_x - padding, text_start_y - padding,
                          text_start_x + max_text_width + padding, text_start_y + text_block_height + padding)
        placed_text_boxes.append(final_text_box)
        cv2.rectangle(output_img, (final_text_box[0], final_text_box[1]),
                      (final_text_box[2], final_text_box[3]), bg_color, -1)
        current_y = text_start_y + line_height_base + padding
        for j, line in enumerate(text_lines):
            cv2.putText(output_img, line, (text_start_x, current_y),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            current_y += total_line_height
    cv2.imwrite("result.jpg", output_img)
    print("Analysis complete. Result saved as result.jpg")
    print("\nObject Data:")
    for i, data in enumerate(shapes_data):
        print(f"\nObject {i+1}:")
        for key, value in data.items():
            print(f"  {key}: {value}")

analyze_shape("image.png")