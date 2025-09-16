
# Hand Gesture Tracker – MVP

Detect three basic hand gestures in real-time using your webcam and control apps via simulated key presses.

## Gestures → Actions
- Open palm: Play/Pause (presses `space`)
- Thumbs up: Volume Up (presses `up`)
- Fist: Next (presses `right`)

These keys work in many apps (YouTube, Spotify desktop with focus, PowerPoint slideshow, etc.). Make sure the target app window is focused.

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. macOS permissions (first run):
- Grant Camera access to your terminal/IDE in System Settings → Privacy & Security → Camera.
- Grant Accessibility access for simulated key presses in System Settings → Privacy & Security → Accessibility (add Terminal/iTerm/Cursor/your IDE).

## Run

```bash
python hand_gesture_tracker.py
```

- Press `q` to quit.
- The window overlays the detected gesture and the last action taken.

## Notes
- Debounce is enabled: the same action will not trigger more than once per second.
- Detection assumes the hand is roughly upright with the palm facing the camera. Thumbs up expects only the thumb extended.
- If gestures are misclassified, improve lighting and ensure your hand is fully visible.

## Troubleshooting
- If the webcam does not open, try changing the index in `cv2.VideoCapture(0)` to `1` or another camera index.
- If no keypress effects occur, ensure the destination app is focused and Accessibility permissions are granted.
- For better reliability on different orientations, consider augmenting the classifier with angles between landmarks.
