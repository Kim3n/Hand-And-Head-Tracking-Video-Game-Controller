import cv2
import time
import threading
import pydirectinput
import mediapipe as mp
import keyboard as kb
from mediapipe.framework.formats import landmark_pb2
from keybinds import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np


#
# MAYBE INSTEAD OF THIS FALLBACK YOU COULD ADD GLOBAL VARIABLE FOR CUSTOM FONT, IF FALSE THEN USE THE NORMAL CV.SETTEXT
# BUT IF TRUE USE THE CUSTOM FONT 
#

try:
    font_path = ""  # For custom font, place a font .ttf in this folder and write the file name => font_path = "Montserrat.tff", leave empty for fall back
    font_path = "FSEX300.ttf"
    # font_path = "Kingthings_Petrock.ttf"
    font = ImageFont.truetype(font_path, 32)  # Adjust font size
except IOError:
    print(f"Warning: '{font_path}' not found. Using default font.")
    font = ImageFont.load_default(size=32)  # Fallback to a system font


# Configuration
pydirectinput.FAILSAFE = False
control_flags = {"hand": False, "head": False, "text": True, "draw": True}
STRAIGHT_VERTICAL_RANGE = (40, 70) # USed for nose deadzone
text_color = (0, 255, 0)
warning_color = (0, 0, 255)



def put_custom_text(cv2_image, text, position, font, color=(255, 255, 255)):
    
    pil_image = Image.fromarray(cv2_image)
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)
    return np.array(pil_image)

class HandTracker:
    def __init__(self):
        self.latest_result = None
        self.last_hand_update = time.time()
        self.frame_lock = threading.Lock()
        
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path='hand_landmarker.task'
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            result_callback=self.update_result
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def update_result(self, result, output_image, timestamp_ms):
        with self.frame_lock:
            self.latest_result = result
            self.last_hand_update = time.time()

    def process_frame(self, frame):
        if control_flags["hand"]:
            small_frame = cv2.resize(frame, (320, 240))
            self.landmarker.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=small_frame),
                int(time.time() * 1000)
            )

    def close(self):
        self.landmarker.close()


class InputHandler:
    def __init__(self):
        self.current_active_keys = set()
        self.last_active_keys = set()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._handle_input, daemon=True)
        self.thread.start()

    def _handle_input(self):
        while self.running:
            with self.lock:
                current_keys = self.current_active_keys.copy()
                last_keys = self.last_active_keys.copy()

            # Release keys that are no longer active
            for key in last_keys - current_keys:
                if key.startswith('mouse click '):
                    parts = key.split(' ')
                    button = parts[2]
                    pydirectinput.mouseUp(button=button)
                else:
                    pydirectinput.keyUp(key)

            # Handle key presses
            for key in current_keys - last_keys:
                if key.startswith('mouse click '):
                    parts = key.split(' ')
                    button = parts[2]
                    pydirectinput.mouseDown(button=button)
                else:
                    pydirectinput.keyDown(key)

            with self.lock:
                self.last_active_keys = current_keys.copy()

            time.sleep(0.05)  # Adjust for responsiveness

    def update_keys(self, keys):
        with self.lock:
            self.current_active_keys = set(keys)

    def stop(self):
        self.running = False
        self.thread.join()
        # Release all keys on exit
        for key in self.last_active_keys:
            pydirectinput.keyUp(key)


def draw_hands(frame, result):
    if not result.hand_landmarks:
        return frame
    
    annotated = frame.copy()
    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
            for lm in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )
        # Get handedness and calculate finger status
        handedness = result.handedness[idx][0].display_name
        display_handedness = "Left" if handedness == "Right" else "Right"  # Swap labels
        finger_status = count_fingers(hand_landmarks, handedness)
        status_text = f"{display_handedness}: {finger_status}"

        (text_width, text_height), _ = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        
        # Get wrist position for text placement
        wrist = hand_landmarks[0]
        h, w = annotated.shape[:2]
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        # Calculate text position (centered below wrist)
        text_x = wrist_x - text_width // 2
        text_y = wrist_y + 35  # 35 pixels below wrist
        
        # Draw finger status text
        # cv2.putText(annotated, f"{display_handedness}: {finger_status}", 
        #             (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.7, text_color, 2)
        annotated = put_custom_text(annotated, f"{display_handedness}: {finger_status}", (text_x, text_y), font, text_color)
                    
    return annotated

import numpy as np

def count_fingers(hand_landmarks, handedness):
    finger_status = [0, 0, 0, 0, 0]

    thumb_tip = hand_landmarks[4]
    thumb_mcp = hand_landmarks[3]

    # 1. Choose the three co-planar landmarks (ideally from the palm)
    points = np.asarray([
        [hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z],
        [hand_landmarks[5].x, hand_landmarks[5].y, hand_landmarks[5].z],
        [hand_landmarks[17].x, hand_landmarks[17].y, hand_landmarks[17].z]
    ])

    # 2. Compute the normal vector (perpendicular to the palm)
    normal_vector = np.cross(points[2] - points[0], points[1] - points[2])
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize

    # 3. Determine hand orientation
    is_vertical = abs(hand_landmarks[5].y - hand_landmarks[17].y) < abs(hand_landmarks[5].x - hand_landmarks[17].x)

    # 4. Check palm direction (towards or away from camera)
    if handedness == "Left":
        palm_facing = normal_vector[2] < 0
    else:  
        palm_facing = normal_vector[2] > 0


    # 5. Finger status logic
    if is_vertical:
        # **For horizontal hands, check z-coordinates**
        for i, fingertip_idx in enumerate([8, 12, 16, 20], start=1):
            tip_y = hand_landmarks[fingertip_idx].y
            dip_y = hand_landmarks[fingertip_idx-1].y
            pip_y = hand_landmarks[fingertip_idx-2].y
            mcp_y = hand_landmarks[fingertip_idx-3].y

            if tip_y < min(dip_y, pip_y, mcp_y):
                finger_status[i] = 1

        # Thumb check for horizontal
        if handedness == 'Right':
            if (palm_facing and thumb_tip.x > thumb_mcp.x) or \
               (not palm_facing  and thumb_tip.x < thumb_mcp.x):
                finger_status[0]= 1
        else:
            if (palm_facing and thumb_tip.x < thumb_mcp.x) or \
               (not palm_facing and thumb_tip.x > thumb_mcp.x):
                finger_status[0] = 1

    else:
        # **For vertical hands, use x-coordinates (except for the thumb)**
        # Thumb logic (still using y-coordinates)
        if handedness == 'Right':
            if (thumb_tip.y < thumb_mcp.y):
                finger_status[0] = 1
            for i, fingertip_idx in enumerate([8, 12, 16, 20], start=1):
                tip_x = hand_landmarks[fingertip_idx].x
                pip_x = hand_landmarks[fingertip_idx - 2].x  

                if (palm_facing and tip_x < pip_x) or \
                (not palm_facing and tip_x > pip_x):
                    finger_status[i] = 1
        else:
            if (thumb_tip.y < thumb_mcp.y):
                finger_status[0] = 1
                    # **For fingers, use x-coordinates instead of y**
            for i, fingertip_idx in enumerate([8, 12, 16, 20], start=1):
                tip_x = hand_landmarks[fingertip_idx].x
                pip_x = hand_landmarks[fingertip_idx - 2].x  

                if (palm_facing and tip_x > pip_x) or \
                (not palm_facing and tip_x < pip_x):
                    finger_status[i] = 1


    return finger_status


class HeadTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        self.last_direction = "Straight"
        self.lock = threading.Lock()
        self.nose_pos = None
        self.left_eye_pos = None
        self.right_eye_pos = None

    def process_frame(self, frame):
        direction = "Straight"
        results = None  # Initialize results variable
        
        if control_flags["head"]:
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results and results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # Extract specific landmarks
            nose = face_landmarks[4]
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]

            nose_pos = (int(nose.x * w), int(nose.y * h))
            left_eye_pos = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_pos = (int(right_eye.x * w), int(right_eye.y * h))

            # Calculate direction
            h_dist = abs(nose_pos[0] - left_eye_pos[0])
            direction = "Left" if h_dist < 20 else "Right" if abs(nose_pos[0] - right_eye_pos[0]) < 20 else "Straight"
            
            v_dist = abs(nose_pos[1] - left_eye_pos[1])
            if v_dist < STRAIGHT_VERTICAL_RANGE[0]:
                direction = "Up"
            elif v_dist > STRAIGHT_VERTICAL_RANGE[1]:
                direction = "Down"

            with self.lock:
                self.last_direction = direction
                self.nose_pos = nose_pos
                self.left_eye_pos = left_eye_pos
                self.right_eye_pos = right_eye_pos
        else:
            with self.lock:
                self.nose_pos = None
                self.left_eye_pos = None
                self.right_eye_pos = None
                self.last_direction = "Straight"

        if control_flags["head"] and direction != "Straight":
            threading.Thread(target=move_mouse, args=(direction,)).start()

    def draw_landmarks(self, frame, y_text, text_color, warning_color):
        with self.lock:
            if control_flags["head"] and control_flags["draw"] and self.nose_pos and self.left_eye_pos and self.right_eye_pos:
                cv2.circle(frame, self.nose_pos, 5, (0, 255, 0), -1)
                cv2.circle(frame, self.left_eye_pos, 5, (0, 0, 255), -1)
                cv2.circle(frame, self.right_eye_pos, 5, (0, 0, 255), -1)
            if control_flags["text"] and control_flags["head"]:
                # cv2.putText(frame, f"Head: {self.last_direction}", 
                #             (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 
                #             0.8, text_color, 2)
                # y_text += 30
                frame = put_custom_text(frame, f"Head: {self.last_direction}",  (10, y_text), font, text_color)
                y_text += 30

                        # Head tracking status
            if not control_flags["head"] and control_flags["text"]:
                # cv2.putText(frame, "Head Tracking: OFF", (10, y_text), 
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)
                # y_text += 30
                frame = put_custom_text(frame, "Head Tracking: OFF", (10, y_text), font, warning_color)
                y_text += 30

        return frame, y_text

    def close(self):
        self.face_mesh.close()

def main():
    hand_tracker = HandTracker()
    head_tracker = HeadTracker()
    input_handler = InputHandler()
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30) # KEEP AT 30, 60 SLOWS THINGS DOWN

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Process tracking threads
        hand_thread = threading.Thread(target=hand_tracker.process_frame, args=(frame,))
        head_thread = threading.Thread(target=head_tracker.process_frame, args=(frame,))
        hand_thread.start()
        head_thread.start()
        
        # Get results
        hand_thread.join(0.01)
        head_thread.join(0.01)

        # Status display
        y_text = 30



        display_frame, y_text = head_tracker.draw_landmarks(display_frame, y_text, text_color, warning_color)
        # Draw overlays if enabled
        if control_flags["draw"]:
            # Draw hand landmarks
            if control_flags["hand"] and hand_tracker.latest_result:
                display_frame = draw_hands(display_frame, hand_tracker.latest_result)
        current_keys = []
        left_hand_text = ""
        right_hand_text = ""
        # Hand tracking status and data
        if control_flags["hand"]:
             # Key state tracking variables

            hand_data = {"Left": None, "Right": None}
            if hand_tracker.latest_result:
                for idx, hand_landmarks in enumerate(hand_tracker.latest_result.hand_landmarks):
                    handedness = hand_tracker.latest_result.handedness[idx][0].display_name
                    display_handedness = "Right" if handedness == "Left" else "Left"
                    finger_status = count_fingers(hand_landmarks, handedness)
                    
                    hand_data[display_handedness] = finger_status     


        
            if hand_data["Left"]:
                left_hand_key, left_hand_text = get_left_hand_key(hand_data['Left'])
                # if left_hand_key:
                #     threading.Thread(target=press_keyboard, args=(left_hand_key,)).start()
                if left_hand_key: current_keys.append(left_hand_key)

                
                
            if hand_data["Right"]:
                right_hand_key, right_hand_text = get_right_hand_key(hand_data['Right'])
                # if right_hand_key:
                #     threading.Thread(target=press_keyboard, args=(right_hand_key,)).start()
                if right_hand_key: current_keys.append(right_hand_key)

            input_handler.update_keys(current_keys)

            if control_flags["text"]:
                if hand_data["Right"]:
                    # cv2.putText(display_frame, f"Right: {right_hand_text}" if right_hand_text else "Right: No detection", (10, y_text), 
                    #         cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    # y_text += 30
                    display_frame = put_custom_text(display_frame, f"Right: {right_hand_text}" if right_hand_text else "Right: No detection", (10, y_text), font, text_color)
                    y_text += 30

                else:
                    # cv2.putText(display_frame, "Right: No detection", (10, y_text), 
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    # y_text += 30
                    display_frame = put_custom_text(display_frame, "Right: No detection", (10, y_text), font, text_color)
                    y_text += 30

                if hand_data["Left"]:
                    # cv2.putText(display_frame, f"Left: {left_hand_text}" if left_hand_text else "Left: No detection", (10, y_text), 
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    # y_text += 30
                    display_frame = put_custom_text(display_frame, f"Left: {left_hand_text}" if left_hand_text else "Left: No detection", (10, y_text), font, text_color)
                    y_text += 30
                else:
                    # cv2.putText(display_frame, f"Left: No detection", (10, y_text), 
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    display_frame = put_custom_text(display_frame, "Left: No detection", (10, y_text), font, text_color)
                    y_text += 30
                    

        elif control_flags["text"]:
                
                # cv2.putText(display_frame, "Hand Tracking: OFF", (10, y_text), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)
                display_frame = put_custom_text(display_frame, "Hand Tracking: OFF", (10, y_text), font, warning_color)
                y_text += 30        
                

        cv2.imshow("Controller", display_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    input_handler.stop()        
    hand_tracker.close()
    head_tracker.close()
    cap.release()
    cv2.destroyAllWindows()

def press_keyboard(key):
    pydirectinput.press(key)


def move_mouse(direction):
    moves = {
        "Left": (-30, 0), "Right": (30, 0),
        "Up": (0, -30), "Down": (0, 30)
    }
    if direction in moves:
        pydirectinput.moveRel(*moves[direction], relative=True)

if __name__ == "__main__":
    kb.add_hotkey('ø', lambda: control_flags.update({"draw": not control_flags["draw"]}))
    kb.add_hotkey('æ', lambda: control_flags.update({"hand": not control_flags["hand"]}))
    kb.add_hotkey('å', lambda: control_flags.update({"head": not control_flags["head"]}))
    kb.add_hotkey('p', lambda: control_flags.update({"text": not control_flags["text"]}))
    main()

