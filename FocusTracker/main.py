import cv2
import mediapipe as mp
import time
import numpy as np
import os
from plyer import notification


SECONDS_TO_ALERT = 2  
YAW_RANGE = 30         
PITCH_RANGE = 25       
DAYS_TO_KEEP = 3       


def cleanup_old_screenshots(folder, days):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return
    
    now = time.time()
    cutoff = now - (days * 86400)
    
    files = os.listdir(folder)
    deleted_count = 0
    for f in files:
        f_path = os.path.join(folder, f)
        if os.path.isfile(f_path):
            if os.path.getmtime(f_path) < cutoff:
                os.remove(f_path)
                deleted_count += 1
    if deleted_count > 0:
        print(f"Очищення: видалено {deleted_count} старих скріншотів.")


cleanup_old_screenshots('Distractions', DAYS_TO_KEEP)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)


start_away_time = None
is_alerted = False  
base_pitch, base_yaw = 0, 0
base_gaze, base_eye_open = 0.012, 0.010 
curr_pitch, curr_yaw, curr_gaze, curr_eye_open = 0, 0, 0, 0
total_focus_start = time.time()

print("СИСТЕМА ГОТОВА. Натисни 'C' для калібрування.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    focused = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_2d, face_3d = [], []
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
            _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1)))
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))[6]
            
            curr_pitch = angles.flatten()[0] * 360
            curr_yaw = angles.flatten()[1] * 360
            diff_pitch = curr_pitch - base_pitch
            diff_yaw = curr_yaw - base_yaw

            pupil = face_landmarks.landmark[468]
            eye_left = face_landmarks.landmark[33]
            eye_right = face_landmarks.landmark[133]
            eye_top = face_landmarks.landmark[159]
            eye_bottom = face_landmarks.landmark[145]

            g_x = pupil.x - (eye_left.x + eye_right.x) / 2
            g_y = pupil.y - (eye_top.y + eye_bottom.y) / 2
            curr_gaze = np.sqrt(g_x**2 + g_y**2)
            curr_eye_open = pupil.y - eye_top.y 

            
            if abs(diff_yaw) < YAW_RANGE and abs(diff_pitch) < PITCH_RANGE:
                if curr_gaze < (base_gaze + 0.010) and curr_eye_open > (base_eye_open - 0.005):
                    focused = True

    current_time = time.time()
    work_duration = int(current_time - total_focus_start)
    timer_text = time.strftime('%H:%M:%S', time.gmtime(work_duration))
    cv2.putText(frame, f"Focus Session: {timer_text}", (img_w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    
    if not focused:
        if start_away_time is None: 
            start_away_time = current_time
        
        elapsed = int(current_time - start_away_time)
        cv2.putText(frame, f"STATUS: AWAY ({elapsed}s)", (10, 50), 0, 1, (0, 0, 255), 3)
        
        if elapsed >= SECONDS_TO_ALERT and not is_alerted:
            time_str = time.strftime('%H.%M.%S')
            file_name = f"Distraction-{time_str}.jpg"
            file_path = os.path.join('Distractions', file_name)
            
            cv2.imwrite(file_path, frame)
            print(f"Збережено: {file_name}")
            
            try: notification.notify(title='Focus Tracker', message=f'Відволікання зафіксовано: {file_name}')
            except: pass
            is_alerted = True 
    else:
        start_away_time = None
        is_alerted = False 


    color = (0, 255, 0) if focused else (0, 0, 255)
    cv2.putText(frame, f"Y:{int(diff_yaw)} P:{int(diff_pitch)} G:{curr_gaze:.3f} E:{curr_eye_open:.3f}", 
                (10, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Pro Focus Tracker', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('й'), 202, 218]: break
    if key in [ord('c'), ord('с'), 241, 242]:
        base_pitch, base_yaw = curr_pitch, curr_yaw
        base_gaze, base_eye_open = curr_gaze, curr_eye_open
        total_focus_start = time.time()
        print("Систему перекалібровано!")

cap.release()
cv2.destroyAllWindows()