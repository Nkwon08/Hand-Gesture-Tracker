import time
from collections import deque, Counter
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import pyautogui


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


Gesture = str


def _dist(a, b) -> float:
	return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _angle(a, b, c) -> float:
	import math
	ba = (a[0] - b[0], a[1] - b[1])
	bc = (c[0] - b[0], c[1] - b[1])
	ba_len = (ba[0] ** 2 + ba[1] ** 2) ** 0.5
	bc_len = (bc[0] ** 2 + bc[1] ** 2) ** 0.5
	if ba_len == 0 or bc_len == 0:
		return 0.0
	cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (ba_len * bc_len)
	cosine = max(-1.0, min(1.0, cosine))
	return math.degrees(math.acos(cosine))


def compute_finger_states(landmarks, image_width: int, image_height: int, is_right: bool) -> Dict[str, bool]:

	px = [
		(int(lm.x * image_width), int(lm.y * image_height))
		for lm in landmarks
	]

	palm_size = max(1.0, _dist(px[5], px[17]))
	wr = px[0]

	def finger_extended(mcp, pip, dip, tip) -> bool:
		pip_angle = _angle(mcp, pip, dip)  
		vertical_check = tip[1] < pip[1] - 5
		return pip_angle > 160 and vertical_check

	index_extended = finger_extended(px[5], px[6], px[7], px[8])
	middle_extended = finger_extended(px[9], px[10], px[11], px[12])
	ring_extended = finger_extended(px[13], px[14], px[15], px[16])
	pinky_extended = finger_extended(px[17], px[18], px[19], px[20])

	thumb_outward = _dist(px[4], wr) / palm_size > 0.8
	thumb_ip_angle = _angle(px[2], px[3], px[4])  
	thumb_extended = thumb_outward and thumb_ip_angle > 150

	return {
		"thumb": thumb_extended,
		"index": index_extended,
		"middle": middle_extended,
		"ring": ring_extended,
		"pinky": pinky_extended,
	}


def classify_gesture(fingers: Dict[str, bool], px, is_right: bool) -> Gesture:
	
	wr = px[0]
	palm_size = max(1.0, _dist(px[5], px[17]))

	extended_count = sum(1 for k in ("index", "middle", "ring", "pinky") if fingers[k])

	fingertips = [px[8], px[12], px[16], px[20]]
	avg_tip_dist = sum(_dist(t, wr) for t in fingertips) / (4 * palm_size)
	if extended_count <= 1 and avg_tip_dist < 0.8:
		return "FIST"

	if extended_count >= 3:
		far_count = sum(1 for t in fingertips if _dist(t, wr) / palm_size > 1.1)
		if far_count >= 3:
			return "OPEN_PALM"

	thumb_up = False
	if fingers["thumb"] and extended_count == 0:
		thumb_tip = px[4]
		thumb_mcp = px[2]
		thumb_up = (thumb_tip[1] + 10) < thumb_mcp[1]  
		thumb_dx = abs(thumb_tip[0] - thumb_mcp[0])
		thumb_dy = abs(thumb_mcp[1] - thumb_tip[1])
		if thumb_up and thumb_dy > thumb_dx:
			return "THUMBS_UP"

	return "UNKNOWN"


def gesture_to_action(gesture: Gesture) -> Optional[Tuple[str, str]]:
	if gesture == "OPEN_PALM":
		return ("Play/Pause", "space")
	if gesture == "THUMBS_UP":
		return ("Volume Up", "up")
	if gesture == "FIST":
		return ("Next", "right")
	return None


def perform_action(key: str) -> None:
	
	pyautogui.press(key)


def draw_overlay(frame, text_left: str, text_right: str) -> None:
	cv2.rectangle(frame, (10, 10), (310, 70), (0, 0, 0), -1)
	cv2.putText(frame, text_left, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
	width = frame.shape[1]
	cv2.rectangle(frame, (width - 310, 10), (width - 10, 70), (0, 0, 0), -1)
	cv2.putText(frame, text_right, (width - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
	pyautogui.FAILSAFE = False

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open webcam (index 0)")

	hands = mp_hands.Hands(
		static_image_mode=False,
		max_num_hands=1,
		model_complexity=1,
		min_detection_confidence=0.6,
		min_tracking_confidence=0.6,
	)

	last_gesture: Gesture = "UNKNOWN"
	stable_frames = 0
	frames_required = 6  
	last_action_time = 0.0
	action_cooldown = 1.0  
	last_action_label = ""

	gesture_window = deque(maxlen=12)

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			result = hands.process(rgb)

			current_gesture: Gesture = "UNKNOWN"
			is_right = True
			if result.multi_hand_landmarks:
				landmarks = result.multi_hand_landmarks[0]
				if result.multi_handedness and len(result.multi_handedness) > 0:
					label = result.multi_handedness[0].classification[0].label
					is_right = label == "Right"

				finger_states = compute_finger_states(landmarks.landmark, frame.shape[1], frame.shape[0], is_right)
				px = [
					(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
					for lm in landmarks.landmark
				]
				current_gesture = classify_gesture(finger_states, px, is_right)

				mp_drawing.draw_landmarks(
					frame,
					landmarks,
					mp_hands.HAND_CONNECTIONS,
					mp_styles.get_default_hand_landmarks_style(),
					mp_styles.get_default_hand_connections_style(),
				)

			gesture_window.append(current_gesture)
			major, count = Counter(g for g in gesture_window if g != "UNKNOWN").most_common(1)[0] if any(g != "UNKNOWN" for g in gesture_window) else ("UNKNOWN", 0)
			final_gesture = major if count >= max(4, int(0.6 * len(gesture_window))) else "UNKNOWN"

			if final_gesture == last_gesture and final_gesture != "UNKNOWN":
				stable_frames += 1
			else:
				stable_frames = 0
				last_gesture = final_gesture

			action_text = last_action_label if last_action_label else "No action"
			if last_gesture in ("OPEN_PALM", "THUMBS_UP", "FIST") and stable_frames >= frames_required:
				mapping = gesture_to_action(last_gesture)
				if mapping is not None:
					label, key = mapping
					now = time.time()
					if now - last_action_time >= action_cooldown:
						perform_action(key)
						last_action_label = f"{label} ({key})"
						last_action_time = now

			gesture_label = last_gesture if last_gesture != "UNKNOWN" else "Detecting..."
			draw_overlay(frame, f"Gesture: {gesture_label}", f"Action: {action_text}")

			cv2.imshow("Hand Gesture Tracker (q to quit)", frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break

	finally:
		hands.close()
		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


