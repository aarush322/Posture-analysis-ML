print("Script started!")
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import asyncio
import requests
import tempfile
import os


print("All imports done!")

# Test webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
else:
    print("Webcam opened successfully!")
cap.release()

# Test model loading
try:
    from ultralytics import YOLO
    yolo = YOLO('yolov8n-pose.pt')
    print("YOLO loaded!")
except Exception as e:
    print(f"YOLO error: {e}")

try:
    model = torch.load('best_model_multiclass.pth', map_location='cpu')
    print("LSTM loaded!")
except Exception as e:
    print(f"LSTM error: {e}")

input("Press Enter to exit...")

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────
BOT_TOKEN   = '8594635121:AAFtc0yayOB07yubBa-oGotCCkxnDRvxOvQ'
CHAT_ID     = '1674783463'
MODEL_PATH  = "best_model_multiclass.pth"
WINDOW_SIZE = 30
COOLDOWN    = 10  # seconds between alerts (avoid spam)

CLASS_NAMES = ['Normal', 'Punching', 'Kicking', 
               'Pushing', 'Sword/Stab', 'Shooting']
SUSPICIOUS_CLASSES = [1, 2, 3, 4, 5]


# ─────────────────────────────────────
# 2. LOAD MODELS
# ─────────────────────────────────────
print('Loading models...')
yolo_model = YOLO('yolov8n-pose.pt')

class PostureLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super(PostureLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=34,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_frame  = lstm_out[:, -1, :]
        output      = self.classifier(last_frame)
        return output

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = PostureLSTM(num_classes=6).to(device)
lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
lstm_model.eval()
print(f'Models loaded on {device}!')


# ─────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────
def extract_keypoints(frame):
    results = yolo_model(frame, verbose=False)
    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        kp = results[0].keypoints.xy[0].cpu().numpy()
        if len(kp) < 17:
            kp = np.zeros((17, 2))
    else:
        kp = np.zeros((17, 2))
    return kp


def normalize_keypoints(keypoints):
    normalized = []
    for frame in keypoints:
        left_hip     = frame[11]
        right_hip    = frame[12]
        center       = (left_hip + right_hip) / 2
        left_sh      = frame[5]
        right_sh     = frame[6]
        shoulder_mid = (left_sh + right_sh) / 2
        torso_length = np.linalg.norm(shoulder_mid - center)
        if torso_length < 1e-6:
            torso_length = 1.0
        norm_frame = (frame - center) / torso_length
        normalized.append(norm_frame)
    return np.array(normalized)


def classify_sequence(keypoints_sequence):
    normalized = normalize_keypoints(keypoints_sequence)
    flattened  = normalized.reshape(30, 34)
    tensor     = torch.FloatTensor(flattened).unsqueeze(0).to(device)
    with torch.no_grad():
        output     = lstm_model(tensor)
        probs      = torch.softmax(output, dim=1)
        confidence = probs.max().item()
        prediction = torch.argmax(output, dim=1).item()
    return prediction, confidence


def send_telegram_alert(label, confidence, screenshot):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Build message
    message = (
        f'🚨 SUSPICIOUS ACTIVITY DETECTED\n\n'
        f'🎯 Activity   : {label}\n'
        f'📊 Confidence : {confidence*100:.1f}%\n'
        f'🕐 Time       : {timestamp}\n\n'
        f'⚠️ Immediate attention required!'
    )

    # Save screenshot temporarily
    temp_path = tempfile.mktemp(suffix='.jpg')
    cv2.imwrite(temp_path, screenshot)

    # Send photo with caption
    url   = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
    with open(temp_path, 'rb') as photo:
        requests.post(url, data={
            'chat_id': CHAT_ID,
            'caption': message
        }, files={'photo': photo})

    # Cleanup temp file
    os.remove(temp_path)
    print(f'[ALERT SENT] {label} ({confidence*100:.1f}%)')


def send_startup_message():
    message = (
        '🟢 Surveillance System Started!\n\n'
        '📷 Webcam is now active\n'
        '🔍 Monitoring for suspicious activities\n'
        '⏰ Running continuously 24/7\n\n'
        '🛡️ Protected classes:\n'
        '   👊 Punching/Boxing\n'
        '   🦵 Kicking\n'
        '   👐 Pushing\n'
        '   🗡️ Sword/Stabbing\n'
        '   🔫 Shooting Stance'
    )
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    requests.post(url, data={'chat_id': CHAT_ID, 'text': message})


# ─────────────────────────────────────
# 4. MAIN SURVEILLANCE LOOP
# ─────────────────────────────────────
def run_surveillance():
    print('Starting webcam surveillance...')

    cap          = cv2.VideoCapture(0)  # 0 = default webcam
    frame_buffer = deque(maxlen=WINDOW_SIZE)
    last_alert   = 0  # timestamp of last alert

    if not cap.isOpened():
        print('ERROR: Cannot open webcam!')
        return

    # Send startup message to Telegram
    send_startup_message()
    print('Surveillance running! Press Q to quit.')

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('ERROR: Cannot read frame!')
            break

        frame_count += 1

        # Extract keypoints
        kp = extract_keypoints(frame)
        frame_buffer.append(kp)

        # Only classify when buffer is full
        if len(frame_buffer) == WINDOW_SIZE:
            sequence           = np.array(list(frame_buffer))
            prediction, confidence = classify_sequence(sequence)

            label = CLASS_NAMES[prediction]

            # Display on screen
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            cv2.putText(frame, f'{label} ({confidence*100:.1f}%)',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 2)

            # Send alert if suspicious and cooldown passed
            current_time = datetime.now().timestamp()
            if (prediction in SUSPICIOUS_CLASSES and
                confidence > 0.80 and
                current_time - last_alert > COOLDOWN):

                # Send alert with screenshot
                send_telegram_alert(label, confidence, frame.copy())
                last_alert = current_time

        # Show live feed
        cv2.imshow('Surveillance Feed', frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Surveillance stopped by user!')
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────
if __name__ == '__main__':
    run_surveillance()
