import cv2
import threading
import csv
import smtplib
from datetime import datetime
from email.message import EmailMessage
import matplotlib.pyplot as plt
import numpy as np
from twilio.rest import Client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==== TWILIO CONFIGURATION ====
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "your twillio account SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "your twillio token id")
TWILIO_FROM = os.getenv("TWILIO_FROM", "your twillio number")
TWILIO_TO = os.getenv("TWILIO_TO", "recive SMS number")

# ==== CONFIGURATION ====
VIDEO_PATH = "crowed.mp4"   # Change this to your video file path
OUTPUT_PATH = "output_counted.mp4"
CSV_PATH = "crowd_log.csv"

WARNING_LIMIT = 150
DANGER_LIMIT = 200

# Sound settings
WARNING_FREQ = 400
DANGER_FREQ = 1000
WARNING_DURATION = 0.5
DANGER_DURATION = 0.3
WARNING_INTERVAL = 1.0
DANGER_INTERVAL = 0.5

# Preview settings
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540
ALERT_FONT_SCALE = 1.2
ALERT_THICKNESS = 2

# Entry zone coordinates
ZONE_TOP_LEFT = (100, 100)
ZONE_BOTTOM_RIGHT = (300, 300)

# ==== EMAIL CONFIGURATION ====
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "your sending massage mail")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your mail pass-key password")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "your massage recive mail")

# ==== Initialize sound function ====
try:
    import winsound
    def play_beep(frequency, duration):
        winsound.Beep(frequency, int(duration * 1000))
except ImportError:
    def play_beep(frequency, duration):
        print(f"Beep {frequency}Hz for {duration}s")

stop_sound = False
sound_thread = None

def play_sound_loop(frequency, duration, interval):
    global stop_sound
    while not stop_sound:
        play_beep(frequency, duration)
        for _ in range(int(interval * 10)):
            if stop_sound:
                break
            cv2.waitKey(100)

def send_email_alert(total_people, alert_level):
    msg = EmailMessage()
    msg.set_content(f"Alert Level: {alert_level}\nTotal People Count: {total_people}\nPlease take necessary action.")
    msg['Subject'] = f"Crowd Alert - {alert_level}"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECEIVER_EMAIL
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"✅ Email alert sent for {alert_level}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def send_sms_alert(total_people, alert_level):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Alert Level: {alert_level}\nTotal People Count: {total_people}",
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        print(f"✅ SMS alert sent: {message.sid}")
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

def detect_mask(face_img):
    # Placeholder for mask detection logic
    return np.random.choice([True, False])

# ==== Video setup ====
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ==== CSV logging setup ====
csv_file = open(CSV_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Total People", "Entered Zone", "Alert Level"])

# ==== Graph setup ====
plt.ion()
fig, ax = plt.subplots()
crowd_counts = []
timestamps = []

# ==== Heatmap setup ====
heatmap = np.zeros((height, width), np.float32)

# ==== Tracking variables ====
seen_ids = set()
total_people = 0
enter_count = 0
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_people = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            detected_people.append((x, y, w, h))

    current_ids = set(range(len(detected_people)))
    for pid in current_ids:
        if pid not in seen_ids:
            seen_ids.add(pid)
            total_people += 1

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    zone_entered = 0
    mask_count = 0
    for i, (x, y, w, h) in enumerate(detected_people):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"ID {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cx = x + w//2
        cy = y + h//2
        if ZONE_TOP_LEFT[0] < cx < ZONE_BOTTOM_RIGHT[0] and ZONE_TOP_LEFT[1] < cy < ZONE_BOTTOM_RIGHT[1]:
            zone_entered += 1
            cv2.putText(frame, "Entered Zone", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        face_img = frame[y:y+h, x:x+w]
        if detect_mask(face_img):
            mask_count += 1
            cv2.putText(frame, "Mask", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # Update heatmap
        heatmap[y:y+h, x:x+w] += 1

    enter_count += zone_entered

    cv2.rectangle(frame, ZONE_TOP_LEFT, ZONE_BOTTOM_RIGHT, (255,0,0), 2)
    cv2.putText(frame, f"Entered: {enter_count}", (ZONE_TOP_LEFT[0], ZONE_TOP_LEFT[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.putText(frame, f"Total Unique People: {total_people}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Masks Detected: {mask_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Time: {now}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Alerts
    alert_text = ""
    alert_color = (0,200,0)
    if sound_thread and sound_thread.is_alive():
        stop_sound = True
        sound_thread.join()
    stop_sound = False
    sound_thread = None

    if total_people > DANGER_LIMIT:
        alert_text = "🚨 DANGER: Too Crowded!"
        alert_color = (0,0,255)
        sound_thread = threading.Thread(target=play_sound_loop, args=(DANGER_FREQ,DANGER_DURATION,DANGER_INTERVAL), daemon=True)
        sound_thread.start()
        threading.Thread(target=send_email_alert, args=(total_people, "Danger"), daemon=True).start()
        threading.Thread(target=send_sms_alert, args=(total_people, "Danger"), daemon=True).start()
    elif total_people > WARNING_LIMIT:
        alert_text = "⚠ WARNING: High Crowd Density"
        alert_color = (0,255,255)
        sound_thread = threading.Thread(target=play_sound_loop, args=(WARNING_FREQ,WARNING_DURATION,WARNING_INTERVAL), daemon=True)
        sound_thread.start()
        threading.Thread(target=send_email_alert, args=(total_people, "Warning"), daemon=True).start()
        threading.Thread(target=send_sms_alert, args=(total_people, "Warning"), daemon=True).start()

    if alert_text:
        cv2.putText(frame, alert_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, ALERT_FONT_SCALE, alert_color, ALERT_THICKNESS, cv2.LINE_AA)

    alert_level = "None"
    if total_people > DANGER_LIMIT:
        alert_level = "Danger"
    elif total_people > WARNING_LIMIT:
        alert_level = "Warning"

    csv_writer.writerow([now, total_people, zone_entered, alert_level])

    # Heatmap visualization - Separate window
    heatmap_display = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_display = np.uint8(heatmap_display)
    heatmap_display = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap_display, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
    cv2.imshow("Crowd Heatmap", heatmap_resized)

    # Update graph
    crowd_counts.append(total_people)
    timestamps.append(now)
    if len(timestamps) > 20:
        timestamps = timestamps[-20:]
        crowd_counts = crowd_counts[-20:]
    ax.clear()
    ax.plot(timestamps, crowd_counts, marker='o')
    ax.set_title("Crowd Count Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total People")
    plt.xticks(rotation=45)
    plt.pause(0.1)

    # Show video preview
    preview = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
    cv2.imshow("People Counter with Safety Alerts", preview)

    if cv2.getWindowProperty("People Counter with Safety Alerts", cv2.WND_PROP_VISIBLE) < 1 or \
       cv2.getWindowProperty("Crowd Heatmap", cv2.WND_PROP_VISIBLE) < 1:
        stop_sound = True
        break
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        stop_sound = True
        break

# Cleanup
cap.release()
out.release()
csv_file.close()
plt.ioff()
cv2.destroyAllWindows()
stop_sound = True

print(f"✅ Done! Output saved as {OUTPUT_PATH}")
print(f"✅ CSV log saved as {CSV_PATH}")

