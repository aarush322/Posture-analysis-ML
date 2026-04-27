# ------- FINAL MODEL ---------
# ---- Run this code after mounting Drive and excecuting the above cell


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler
import tempfile
from datetime import datetime

# ─────────────────────────────────────
# 1. LOAD MODELS
# ─────────────────────────────────────
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

# Class names
CLASS_NAMES = ['Normal', 'Punching', 'Kicking', 'Pushing', 'Sword/Stab', 'Shooting']
SUSPICIOUS_CLASSES = [1, 2, 3, 4, 5]  # anything not normal

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model = PostureLSTM(num_classes=6).to(device)
lstm_model.load_state_dict(torch.load(
    '/content/drive/MyDrive/Posture_Dataset/best_model_multiclass.pth',
    map_location=device))
lstm_model.eval()
print(f'Models loaded on {device}!')


# ─────────────────────────────────────
# 2. HELPER FUNCTIONS
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
    # keypoints_sequence shape: (30, 17, 2)
    normalized = normalize_keypoints(keypoints_sequence)
    flattened  = normalized.reshape(30, 34)
    tensor     = torch.FloatTensor(flattened).unsqueeze(0).to(device)
    with torch.no_grad():
        output     = lstm_model(tensor)
        probs      = torch.softmax(output, dim=1)
        confidence = probs.max().item()
        prediction = torch.argmax(output, dim=1).item()
    return prediction, confidence


def analyze_all_windows(all_keypoints):
    # Analyze every 30 frame window
    # Returns most suspicious result

    total_frames     = len(all_keypoints)
    best_prediction  = 0  # default normal
    best_confidence  = 0.0
    best_label       = 'Normal'

    for start in range(0, total_frames - 30 + 1, 1):
        end      = start + 30
        sequence = np.array(all_keypoints[start:end])

        prediction, confidence = classify_sequence(sequence)

        # If suspicious and more confident than before
        if prediction in SUSPICIOUS_CLASSES and confidence > best_confidence:
            best_prediction = prediction
            best_confidence = confidence
            best_label      = CLASS_NAMES[prediction]

    # If no suspicious window found → return most confident normal
    if best_prediction == 0:
        best_label      = 'Normal'
        best_confidence = 0.80

    return best_label, best_confidence


def build_response(label, confidence, mode):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if label != 'Normal':
        return (
            f'🚨 SUSPICIOUS ACTIVITY DETECTED\n\n'
            f'📌 Mode       : {mode}\n'
            f'🎯 Activity   : {label}\n'
            f'📊 Confidence : {confidence*100:.1f}%\n'
            f'🕐 Time       : {timestamp}\n\n'
            f'⚠️ Immediate attention required!'
        )
    else:
        return (
            f'✅ NORMAL ACTIVITY\n\n'
            f'📌 Mode       : {mode}\n'
            f'🎯 Activity   : {label}\n'
            f'📊 Confidence : {confidence*100:.1f}%\n'
            f'🕐 Time       : {timestamp}\n\n'
            f'No threat detected.'
        )


# ─────────────────────────────────────
# 3. TELEGRAM HANDLERS
# ─────────────────────────────────────
BOT_TOKEN = 'PASTE_YOUR_BOT_TOKEN_HERE'  # ← paste token here

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        '👋 Welcome to Surveillance Posture Bot!\n\n'
        '📸 Send a PHOTO → instant analysis\n'
        '🎥 Send a VIDEO → full LSTM analysis\n\n'
        '🔍 Detects:\n'
        '   👊 Punching/Boxing\n'
        '   🦵 Kicking\n'
        '   👐 Pushing\n'
        '   🗡️ Sword/Stabbing\n'
        '   🔫 Shooting stance\n'
        '   ✅ Normal activity\n'
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('📸 Photo received! Analyzing...')
    try:
        # Download photo
        photo = await update.message.photo[-1].get_file()
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            await photo.download_to_drive(f.name)
            frame = cv2.imread(f.name)

        # Extract keypoints from single frame
        kp = extract_keypoints(frame)

        # Duplicate frame 30 times → feed to LSTM
        sequence           = np.array([kp] * 30)  # (30, 17, 2)
        prediction, confidence = classify_sequence(sequence)
        label              = CLASS_NAMES[prediction]

        response = build_response(label, confidence, 'Photo')
        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f'❌ Error: {str(e)}')


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('🎥 Video received! Analyzing all frames...')
    try:
        # Download video
        video = await update.message.video.get_file()
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            await video.download_to_drive(f.name)
            video_path = f.name

        # Extract ALL keypoints
        cap           = cv2.VideoCapture(video_path)
        all_keypoints = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kp = extract_keypoints(frame)
            all_keypoints.append(kp)
        cap.release()

        # Check minimum frames
        if len(all_keypoints) < 30:
            await update.message.reply_text(
                '⚠️ Video too short! Please send at least 1 second.')
            return

        await update.message.reply_text(
            f'🔍 Analyzing {len(all_keypoints)} frames...')

        # Analyze ALL windows → find most suspicious
        label, confidence = analyze_all_windows(all_keypoints)

        response = build_response(label, confidence, 'Video')
        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f'❌ Error: {str(e)}')


# ─────────────────────────────────────
# 4. RUN BOT
# ─────────────────────────────────────
print('Starting bot...')
app = Application.builder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler('start', start))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
app.add_handler(MessageHandler(filters.VIDEO, handle_video))

print('Bot is running!')
await app.initialize()
await app.start()
await app.updater.start_polling()
