import streamlit as st
import requests
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from streamlit_autorefresh import st_autorefresh
import cv2
import torch
import time
import datetime
from PIL import Image
import numpy as np
import io
import os
from telegram.ext import Application
from telegram import Bot

# Atur cache PyTorch untuk lingkungan deployment
os.environ['TORCH_HOME'] = '/tmp/torch_hub'

# --- CONFIG ---
UBIDOTS_TOKEN = "BBUS-JBKLQqTfq2CPXNytxeUfSaTjekeL1K"
DEVICE_LABEL = "hsc345"
VARIABLES = ["mq2", "humidity", "temperature", "lux"]
TELEGRAM_BOT_TOKEN = "7941979379:AAEWGtlb87RYkvht8GzL8Ber29uosKo3e4s"
TELEGRAM_CHAT_ID = "5721363432"
NOTIFICATION_INTERVAL = 300  # 5 menit dalam detik
ALERT_COOLDOWN = 60  # 1 menit cooldown untuk notifikasi langsung

# --- STYLE ---
st.markdown("""
    <style>
        .main-title {
            background-color: #001f3f;
            color: white;
            padding: 15px;
            text-align: center;
            font-size: 32px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        .data-box {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            margin-bottom: 10px;
            font-size: 22px;
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
        }
        .label {
            font-weight: bold;
        }
        .data-value {
            font-size: 24px;
            font-weight: bold;
        }
        .refresh-btn {
            position: absolute;
            top: 30px;
            right: 30px;
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .refresh-btn:hover {
            background-color: #005a8d;
        }
        .tab-content {
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- TELEGRAM FUNCTIONS ---
async def send_telegram_message(message):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="Markdown")
        st.success("Pesan berhasil dikirim ke Telegram!")
    except Exception as e:
        st.error(f"Gagal mengirim pesan ke Telegram: {str(e)}")

async def send_telegram_photo(photo, caption):
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo, caption=caption, parse_mode="Markdown")
        st.success("Foto berhasil dikirim ke Telegram!")
    except Exception as e:
        st.error(f"Gagal mengirim foto ke Telegram: {str(e)}")

# --- DATA FETCH ---
def get_ubidots_data(variable_label):
    url = f"https://industrial.api.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}/{variable_label}/values"
    headers = {
        "X-Auth-Token": UBIDOTS_TOKEN,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            st.error(f"Gagal mengambil data dari Ubidots untuk {variable_label}: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error saat mengambil data Ubidots: {str(e)}")
        return None

# --- SIMULASI DATA DAN MODEL ---
@st.cache_data
def generate_mq2_simulation_data(n_samples=100):
    data = []
    for _ in range(n_samples):
        label = random.choices([0, 1], weights=[0.7, 0.3])[0]
        value = random.randint(400, 1000) if label == 1 else random.randint(100, 400)
        data.append((value, label))
    df = pd.DataFrame(data, columns=["mq2_value", "label"])
    return df

@st.cache_resource
def train_mq2_model():
    df = generate_mq2_simulation_data()
    X = df[['mq2_value']]
    y = df['label']
    model = LogisticRegression()
    model.fit(X, y)
    return model

model_iot = train_mq2_model()

# --- AI LOGIC ---
def predict_smoke_status(mq2_value):
    if mq2_value is None:
        return "Data asap tidak tersedia."
    if mq2_value > 800:
        return "Bahaya! Terdeteksi asap rokok!"
    elif mq2_value >= 500:
        return "Mencurigakan: kemungkinan ada asap, tapi belum pasti rokok."
    else:
        return "Semua aman, tidak terdeteksi asap mencurigakan."

def evaluate_lux_condition(lux_value, mq2_value):
    if lux_value is None:
        return "Data cahaya tidak tersedia."
    if lux_value <= 50:
        if "Bahaya" in predict_smoke_status(mq2_value):
            return "Agak mencurigakan: gelap dan ada indikasi asap rokok!"
        elif "Mencurigakan" in predict_smoke_status(mq2_value):
            return "Ruangan gelap dan ada kemungkinan asap, perlu dipantau."
        else:
            return "Ruangan dalam kondisi gelap, tapi tidak ada asap. Masih aman."
    else:
        return "Lampu menyala, kondisi ruangan terang."

def evaluate_temperature_condition(temp_value):
    if temp_value is None:
        return "Data suhu tidak tersedia."
    if temp_value >= 31:
        return "Suhu sangat panas, bisa tidak nyaman, bisa berbahaya!"
    elif temp_value >= 29:
        return "Suhu cukup panas, kurang nyaman."
    elif temp_value <= 28:
        return "Suhu normal dan nyaman."
    else:
        return "Suhu terlalu dingin, bisa tidak nyaman."

def generate_narrative_report(mq2_status, mq2_value, lux_status, lux_value, temp_status, temp_value, humidity_status, humidity_value):
    intro_templates = [
        "Saat ini, kondisi ruangan terpantau sebagai berikut.",
        "Berikut laporan terbaru dari sistem pengawasan ruangan.",
        "Mari kita lihat bagaimana keadaan ruangan saat ini."
    ]
    smoke_templates = {
        "Bahaya": [
            f"Perhatian! Sensor mendeteksi asap rokok dengan nilai MQ2 {mq2_value}. Segera periksa ruangan!",
            f"Nilai MQ2 mencapai {mq2_value}, menunjukkan adanya asap rokok. Tindakan cepat diperlukan."
        ],
        "Mencurigakan": [
            f"Ada indikasi asap dengan nilai MQ2 {mq2_value}. Meski belum pasti rokok, sebaiknya waspada.",
            f"Sensor MQ2 mencatat {mq2_value}, kemungkinan ada asap. Perlu pemantauan lebih lanjut."
        ],
        "Semua aman": [
            f"Semuanya aman, sensor MQ2 hanya mencatat {mq2_value}. Tidak ada tanda-tanda asap rokok.",
            f"Ruangan bebas asap dengan nilai MQ2 {mq2_value}. Kondisi terkendali."
        ]
    }
    light_templates = {
        "mencurigakan": [
            f"Pencahayaan sangat rendah ({lux_value} lux), ditambah ada indikasi asap. Situasi perlu diperhatikan.",
            f"Dengan lux hanya {lux_value}, ruangan gelap dan ada tanda asap. Sebaiknya periksa."
        ],
        "gelap": [
            f"Ruangan gelap dengan intensitas cahaya {lux_value} lux, tapi tidak ada asap. Aman untuk saat ini.",
            f"Pencahayaan rendah ({lux_value} lux), namun tidak ada masalah asap."
        ],
        "terang": [
            f"Ruangan terang dengan cahaya {lux_value} lux. Semua terlihat jelas dan baik.",
            f"Dengan {lux_value} lux, pencahayaan ruangan sangat mendukung visibilitas."
        ]
    }
    temp_templates = {
        "panas": [
            f"Suhu cukup tinggi, mencapai {temp_value}°C. Mungkin perlu ventilasi tambahan.",
            f"Ruangan terasa panas dengan suhu {temp_value}°C. Perhatikan kenyamanan."
        ],
        "normal": [
            f"Suhu nyaman di {temp_value}°C. Ideal untuk aktivitas sehari-hari.",
            f"Dengan suhu {temp_value}°C, ruangan dalam kondisi menyenangkan."
        ],
        "dingin": [
            f"Suhu agak dingin, hanya {temp_value}°C. Mungkin perlu penghangat.",
            f"Ruangan terasa sejuk di {temp_value}°C. Sesuaikan jika diperlukan."
        ]
    }
    humidity_templates = {
        "tinggi": [
            f"Kelembapan tinggi di {humidity_value}%. Pertimbangkan untuk menggunakan dehumidifier.",
            f"Dengan kelembapan {humidity_value}%, ruangan terasa agak lembap."
        ],
        "normal": [
            f"Kelembapan normal di {humidity_value}%. Kondisi cukup seimbang.",
            f"Level kelembapan {humidity_value}% menunjukkan keseimbangan yang baik."
        ],
        "rendah": [
            f"Kelembapan rendah, hanya {humidity_value}%. Mungkin perlu humidifier.",
            f"Ruangan agak kering dengan kelembapan {humidity_value}%."
        ]
    }

    intro = random.choice(intro_templates)
    smoke = random.choice(smoke_templates.get(mq2_status.split()[0], ["Data asap tidak tersedia."]))
    light_key = "mencurigakan" if "mencurigakan" in lux_status.lower() else "gelap" if "gelap" in lux_status.lower() else "terang"
    light = random.choice(light_templates.get(light_key, ["Data cahaya tidak tersedia."]))
    temp_key = "panas" if "panas" in temp_status.lower() else "dingin" if "dingin" in temp_status.lower() else "normal"
    temp = random.choice(temp_templates.get(temp_key, ["Data suhu tidak tersedia."]))
    humidity_key = "tinggi" if "tinggi" in humidity_status.lower() else "rendah" if "rendah" in humidity_status.lower() else "normal"
    humidity = random.choice(humidity_templates.get(humidity_key, ["Data kelembapan tidak tersedia."]))

    narrative = (
        f"📊 *Laporan Status Ruangan*\n"
        f"{intro}\n"
        f"- 🚨 {smoke}\n"
        f"- 💡 {light}\n"
        f"- 🌡️ {temp}\n"
        f"- 💧 {humidity}\n"
        f"🕒 *Waktu*: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return narrative

def predict_smoking_risk_rule_based(mq2_value, lux_value):
    if mq2_value is None or lux_value is None:
        return "Data tidak cukup untuk prediksi risiko merokok."
    
    risk_score = 0
    risk_messages = []
    
    if mq2_value > 800:
        risk_score += 50
        risk_messages.append("Nilai asap sangat tinggi (MQ2 > 800), besar kemungkinan ada aktivitas merokok.")
    elif mq2_value >= 500:
        risk_score += 30
        risk_messages.append("Asap mencurigakan terdeteksi (MQ2 500-800), mungkin ada risiko merokok.")
    else:
        risk_messages.append("Asap rendah (MQ2 < 500), tidak ada indikasi kuat merokok.")
    
    if lux_value <= 50:
        risk_score += 20
        risk_messages.append("Ruangan gelap (lux ≤ 50), sering dikaitkan dengan aktivitas tersembunyi seperti merokok.")
    elif lux_value <= 100:
        risk_score += 10
        risk_messages.append("Pencahayaan rendah (lux ≤ 100), bisa memudahkan aktivitas merokok tanpa terdeteksi.")
    else:
        risk_messages.append("Ruangan terang (lux > 100), mengurangi kemungkinan merokok tersembunyi.")
    
    if risk_score >= 60:
        risk_level = "Tinggi"
        recommendation = "Segera periksa ruangan dan pastikan ventilasi baik. Aktivitas merokok sangat mungkin terjadi."
    elif risk_score >= 40:
        risk_level = "Sedang"
        recommendation = "Pantau ruangan lebih sering, terutama jika asap tetap terdeteksi. Pertimbangkan pemeriksaan manual."
    else:
        risk_level = "Rendah"
        recommendation = "Kondisi aman untuk saat ini. Tetap pertahankan pemantauan rutin."
    
    report = (
        f"🔍 *Prediksi Risiko Merokok*\n"
        f"Tingkat Risiko: **{risk_level}** (Skor: {risk_score})\n"
        f"Rincian:\n- {risk_messages[0]}\n- {risk_messages[1]}\n"
        f"Rekomendasi: {recommendation}"
    )
    return report

def get_room_condition_summary(mq2_value, lux_value, temperature_value, humidity_value):
    mq2_status = predict_smoke_status(mq2_value)
    lux_status = evaluate_lux_condition(lux_value, mq2_value)
    temp_status = evaluate_temperature_condition(temperature_value)
    humidity_status = (f"Kelembapan {humidity_value}%: {'tinggi' if humidity_value > 70 else 'normal' if humidity_value >= 30 else 'rendah'}"
                      if humidity_value is not None else "Data kelembapan tidak tersedia.")

    if "Bahaya" in mq2_status or "berbahaya" in temp_status.lower():
        overall_status = "Bahaya"
        color = "red"
        suggestion = "Segera ambil tindakan: periksa sumber asap atau atur suhu ruangan."
    elif "Mencurigakan" in mq2_status or "mencurigakan" in lux_status.lower() or "panas" in temp_status.lower() or "tinggi" in humidity_status.lower():
        overall_status = "Waspada"
        color = "orange"
        suggestion = "Pantau ruangan lebih sering dan pertimbangkan ventilasi atau penyesuaian cahaya."
    else:
        overall_status = "Aman"
        color = "green"
        suggestion = "Kondisi ruangan baik, lanjutkan pemantauan rutin."

    return {
        "status": overall_status,
        "color": color,
        "suggestion": suggestion,
        "details": {
            "Asap": mq2_status,
            "Cahaya": lux_status,
            "Suhu": temp_status,
            "Kelembapan": humidity_status
        }
    }

def generative_chatbot_response(question, mq2_value, lux_value, temperature_value, humidity_value):
    question = question.lower().strip()
    mq2_status = predict_smoke_status(mq2_value)
    lux_status = evaluate_lux_condition(lux_value, mq2_value)
    temp_status = evaluate_temperature_condition(temperature_value)
    humidity_status = (f"Kelembapan {humidity_value}%: {'tinggi' if humidity_value > 70 else 'normal' if humidity_value >= 30 else 'rendah'}"
                      if humidity_value is not None else "Data kelembapan tidak tersedia.")

    responses = {
        "asap_rokok": [
            f"Saat ini, {mq2_status.lower()}.",
            f"Kondisi asap: {mq2_status.lower()}."
        ],
        "cahaya": [
            f"Pencahayaan ruangan: {lux_status.lower()}.",
            f"Kondisi cahaya saat ini: {lux_status.lower()}."
        ],
        "suhu": [
            f"Suhu ruangan: {temp_status.lower()}.",
            f"Kondisi suhu saat ini: {temp_status.lower()}."
        ],
        "kelembapan": [
            f"{humidity_status.lower()}.",
            f"Status kelembapan: {humidity_status.lower()}."
        ],
        "status_umum": [
            (
                f"Status ruangan saat ini:\n"
                f"- Asap: {mq2_status.lower()}\n"
                f"- Cahaya: {lux_status.lower()}\n"
                f"- Suhu: {temp_status.lower()}\n"
                f"- Kelembapan: {humidity_status.lower()}"
            ),
            (
                f"Ringkasan kondisi:\n"
                f"- {mq2_status.lower()}\n"
                f"- {lux_status.lower()}\n"
                f"- {temp_status.lower()}\n"
                f"- {humidity_status.lower()}"
            )
        ]
    }

    suggestions = []
    if "Bahaya" in mq2_status:
        suggestions.append("Segera buka jendela untuk ventilasi dan periksa sumber asap.")
    elif "Mencurigakan" in mq2_status:
        suggestions.append("Pantau ruangan lebih sering atau lakukan pemeriksaan manual.")
    if "gelap" in lux_status.lower():
        suggestions.append("Nyalakan lampu untuk meningkatkan visibilitas.")
    if "panas" in temp_status.lower():
        suggestions.append("Pertimbangkan menyalakan AC atau membuka ventilasi.")
    elif "dingin" in temp_status.lower():
        suggestions.append("Gunakan penghangat jika merasa tidak nyaman.")
    if "tinggi" in humidity_status.lower():
        suggestions.append("Gunakan dehumidifier untuk mengurangi kelembapan.")
    elif "rendah" in humidity_status.lower():
        suggestions.append("Pertimbangkan humidifier untuk menambah kelembapan.")

    suggestion_text = "\nSaran: " + " ".join(suggestions) if suggestions else ""

    if any(word in question for word in ["rokok", "asap"]):
        return random.choice(responses["asap_rokok"]) + suggestion_text
    elif any(word in question for word in ["lampu", "cahaya", "gelap", "pencahayaan"]):
        return random.choice(responses["cahaya"]) + suggestion_text
    elif any(word in question for word in ["suhu", "panas", "dingin"]):
        return random.choice(responses["suhu"]) + suggestion_text
    elif any(word in question for word in ["kelembapan", "lembap"]):
        return random.choice(responses["kelembapan"]) + suggestion_text
    elif any(word in question for word in ["status", "kondisi", "keadaan", "deteksi", "bahaya"]):
        return random.choice(responses["status_umum"]) + suggestion_text
    else:
        return (
            "Maaf, saya hanya bisa menjawab tentang kondisi ruangan seperti asap, cahaya, suhu, atau kelembapan. "
            "Coba tanyakan seperti 'Apa status asap?' atau 'Bagaimana suhu di sini?'"
        )

# --- ESP32-CAM DETECTION ---
@st.cache_resource
def load_yolo_model():
    try:
        st.write("Mencoba memuat model YOLOv5...")
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Model file exists: {os.path.exists('model/best.pt')}")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)
        st.write("Model YOLOv5 berhasil dimuat.")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLOv5: {str(e)}")
        return None

def run_camera_detection(frame_placeholder, status_placeholder):
    try:
        cap = cv2.VideoCapture('http://192.168.1.6:81/stream')
        if not cap.isOpened():
            status_placeholder.error("Tidak dapat membuka stream kamera. Periksa URL atau koneksi ESP32-CAM.")
            return
        last_saved_time = 0
        last_smoking_notification = 0
        save_interval = 600  # 10 menit untuk penyimpanan gambar lokal

        while st.session_state.get("cam_running", False) and st.session_state.get("cam_refresh", False):
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Gagal membaca frame dari kamera. Stream mungkin terputus.")
                break

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if st.session_state.get("model_cam") is not None:
                results = st.session_state.model_cam(img_pil)
                results.render()
                rendered = results.ims[0]
                frame = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

                df = results.pandas().xyxy[0]
                found_person = 'person' in df['name'].values
                found_smoke = 'smoke' in df['name'].values
            else:
                found_person = False
                found_smoke = False

            current_time = time.time()

            _, buffer = cv2.imencode('.jpg', frame)
            st.session_state.latest_frame = buffer.tobytes()

            if found_person and found_smoke:
                status_placeholder.warning("Merokok terdeteksi di ruangan!")
                if current_time - last_smoking_notification > ALERT_COOLDOWN:
                    caption = (
                        f"🚨 *Peringatan*: Aktivitas merokok terdeteksi di ruangan!\n"
                        f"🕒 *Waktu*: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    import asyncio
                    asyncio.run(send_telegram_photo(st.session_state.latest_frame, caption))
                    last_smoking_notification = current_time

                    mq2_status = st.session_state.last_notification['mq2']['status'] or "Tidak ada data"
                    mq2_value = st.session_state.last_notification['mq2']['value'] or "N/A"
                    lux_status = st.session_state.last_notification['lux']['status'] or "Tidak ada data"
                    lux_value = st.session_state.last_notification['lux']['value'] or "N/A"
                    temp_status = st.session_state.last_notification['temperature']['status'] or "Tidak ada data"
                    temp_value = st.session_state.last_notification['temperature']['value'] or "N/A"
                    humidity_status = st.session_state.last_notification['humidity']['status'] or "Tidak ada data"
                    humidity_value = st.session_state.last_notification['humidity']['value'] or "N/A"

                    narrative = generate_narrative_report(
                        mq2_status, mq2_value, lux_status, lux_value,
                        temp_status, temp_value, humidity_status, humidity_value
                    )
                    asyncio.run(send_telegram_photo(st.session_state.latest_frame, narrative))

                if current_time - last_saved_time > save_interval:
                    filename = datetime.datetime.now().strftime("smoking_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(filename, frame)
                    last_saved_time = current_time
                    status_placeholder.info(f"Gambar disimpan: {filename}")
            else:
                status_placeholder.success("Tidak ada aktivitas merokok terdeteksi di ruangan.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_placeholder.image(frame_pil, channels="RGB", use_container_width=True)
            st.session_state.last_frame = frame_pil

            time.sleep(0.1)

        cap.release()
    except Exception as e:
        status_placeholder.error(f"Error kamera: {str(e)}")
    finally:
        pass  # Tidak perlu cv2.destroyAllWindows()

# --- PERIODIC NOTIFICATION FUNCTION ---
def send_periodic_notification():
    current_time = time.time()
    if current_time - st.session_state.last_notification['last_sent'] > NOTIFICATION_INTERVAL:
        mq2_status = st.session_state.last_notification['mq2']['status'] or "Tidak ada data"
        mq2_value = st.session_state.last_notification['mq2']['value'] or "N/A"
        lux_status = st.session_state.last_notification['lux']['status'] or "Tidak ada data"
        lux_value = st.session_state.last_notification['lux']['value'] or "N/A"
        temp_status = st.session_state.last_notification['temperature']['status'] or "Tidak ada data"
        temp_value = st.session_state.last_notification['temperature']['value'] or "N/A"
        humidity_status = st.session_state.last_notification['humidity']['status'] or "Tidak ada data"
        humidity_value = st.session_state.last_notification['humidity']['value'] or "N/A"

        caption = generate_narrative_report(
            mq2_status, mq2_value, lux_status, lux_value,
            temp_status, temp_value, humidity_status, humidity_value
        )

        import asyncio
        if st.session_state.latest_frame is not None:
            asyncio.run(send_telegram_photo(st.session_state.latest_frame, caption))
        else:
            asyncio.run(send_telegram_message(caption + "\n⚠️ *Foto*: Kamera tidak aktif"))

        st.session_state.last_notification['last_sent'] = current_time

# --- UI START ---
st.markdown('<div class="main-title">Sistem Deteksi Merokok Terintegrasi</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["IoT Sensor", "ESP32-CAM"])

# --- IOT TAB ---
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.header("Live Stream Data + AI Deteksi Rokok & Cahaya")

    mq2_value_latest = None
    lux_value_latest = None
    temperature_value_latest = None
    humidity_value_latest = None

    auto_refresh_iot = st.checkbox("Aktifkan Auto-Refresh Data IoT", value=True, key="iot_refresh")
    if auto_refresh_iot:
        st_autorefresh(interval=5000, key="iot_auto_refresh")

    if 'last_notification' not in st.session_state:
        st.session_state.last_notification = {
            'mq2': {'status': None, 'value': None, 'last_alert_sent': 0},
            'lux': {'status': None, 'value': None},
            'temperature': {'status': None, 'value': None},
            'humidity': {'status': None, 'value': None},
            'last_sent': 0
        }

    if 'latest_frame' not in st.session_state:
        st.session_state.latest_frame = None

    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None

    for var_name in VARIABLES:
        data = get_ubidots_data(var_name)
        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            value = round(df.iloc[0]['value'], 2)

            if var_name == "mq2":
                var_label = "ASAP/GAS"
                emoji = "💨"
                mq2_value_latest = value
            elif var_name == "humidity":
                var_label = "KELEMBAPAN"
                emoji = "💧"
                humidity_value_latest = value
            elif var_name == "temperature":
                var_label = "SUHU"
                emoji = "🌡️"
                temperature_value_latest = value
            elif var_name == "lux":
                var_label = "INTENSITAS CAHAYA"
                emoji = "💡"
                lux_value_latest = value

            st.markdown(
                f'<div class="data-box"><span class="label">{emoji} {var_label}</span><span class="data-value">{value}</span></div>',
                unsafe_allow_html=True
            )

            st.line_chart(df[['timestamp', 'value']].set_index('timestamp'))

            current_time = time.time()

            if var_name == "mq2":
                status = predict_smoke_status(value)
                if "Bahaya" in status and \
                   current_time - st.session_state.last_notification['mq2']['last_alert_sent'] > ALERT_COOLDOWN:
                    caption = (
                        f"🚨 *Peringatan Asap*: {status}\n"
                        f"📊 *Nilai MQ2*: {value}\n"
                        f"🕒 *Waktu*: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    import asyncio
                    if st.session_state.latest_frame is not None:
                        asyncio.run(send_telegram_photo(st.session_state.latest_frame, caption))
                    else:
                        asyncio.run(send_telegram_message(caption + "\n⚠️ *Foto*: Kamera tidak aktif"))
                    st.session_state.last_notification['mq2']['last_alert_sent'] = current_time

                    narrative = generate_narrative_report(
                        status, value,
                        st.session_state.last_notification['lux']['status'] or "Tidak ada data",
                        st.session_state.last_notification['lux']['value'] or "N/A",
                        st.session_state.last_notification['temperature']['status'] or "Tidak ada data",
                        st.session_state.last_notification['temperature']['value'] or "N/A",
                        st.session_state.last_notification['humidity']['status'] or "Tidak ada data",
                        st.session_state.last_notification['humidity']['value'] or "N/A"
                    )
                    if st.session_state.latest_frame is not None:
                        asyncio.run(send_telegram_photo(st.session_state.latest_frame, narrative))
                    else:
                        asyncio.run(send_telegram_message(narrative + "\n⚠️ *Foto*: Kamera tidak aktif"))

                st.session_state.last_notification['mq2']['status'] = status
                st.session_state.last_notification['mq2']['value'] = value
                if "Bahaya" in status:
                    st.error(status)
                elif "Mencurigakan" in status:
                    st.warning(status)
                else:
                    st.success(status)

            if var_name == "lux":
                lux_status = evaluate_lux_condition(value, mq2_value_latest)
                st.session_state.last_notification['lux']['status'] = lux_status
                st.session_state.last_notification['lux']['value'] = value
                if "mencurigakan" in lux_status.lower():
                    st.warning(lux_status)
                else:
                    st.info(lux_status)

            if var_name == "temperature":
                temp_status = evaluate_temperature_condition(value)
                st.session_state.last_notification['temperature']['status'] = temp_status
                st.session_state.last_notification['temperature']['value'] = value
                if "panas" in temp_status.lower() or "berbahaya" in temp_status.lower():
                    st.warning(temp_status)
                elif "dingin" in temp_status.lower():
                    st.info(temp_status)
                else:
                    st.success(temp_status)

            if var_name == "humidity":
                humidity_status = f"Kelembapan {value}%: {'tinggi' if value > 70 else 'normal' if value >= 30 else 'rendah'}"
                st.session_state.last_notification['humidity']['status'] = humidity_status
                st.session_state.last_notification['humidity']['value'] = value
                st.info(humidity_status)

        else:
            st.error(f"Gagal mengambil data dari variabel: {var_name}")

    if all(v is not None for v in [mq2_value_latest, lux_value_latest, temperature_value_latest, humidity_value_latest]):
        summary = get_room_condition_summary(
            mq2_value_latest, lux_value_latest, temperature_value_latest, humidity_value_latest
        )
        st.markdown(
            f"""
            <div style="background-color: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: {summary['color']};">Status Ruangan: {summary['status']}</h3>
                <p><strong>Saran:</strong> {summary['suggestion']}</p>
                <p><strong>Detail:</strong></p>
                <ul>
                    <li>Asap: {summary['details']['Asap']}</li>
                    <li>Cahaya: {summary['details']['Cahaya']}</li>
                    <li>Suhu: {summary['details']['Suhu']}</li>
                    <li>Kelembapan: {summary['details']['Kelembapan']}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    send_periodic_notification()

    # Tombol uji notifikasi Telegram
    st.subheader("Uji Notifikasi Telegram")
    if st.button("Kirim Pesan Uji ke Telegram"):
        import asyncio
        asyncio.run(send_telegram_message("🔍 *Pesan Uji*: Sistem deteksi merokok berfungsi dengan baik!"))

    if mq2_value_latest is not None:
        st.markdown("---")
        st.subheader("💬 Chatbot Pengawas")
        user_question = st.text_input("Tanyakan sesuatu tentang ruangan:", "", key="chatbot_input")
        if user_question:
            response = generative_chatbot_response(
                user_question,
                mq2_value_latest,
                lux_value_latest,
                temperature_value_latest,
                humidity_value_latest
            )
            st.write(f"🤖: {response}")
        
        st.write("Atau pilih pertanyaan di bawah ini:")
        questions = [
            "Ada asap rokok di sini?",
            "Bagaimana situasi asap rokok?",
            "Apakah terdeteksi asap rokok?",
            "Ada bahaya asap rokok?",
            "Status umum di sekitar?",
            "Apa status cahaya di ruangan?",
            "Bagaimana kondisi cahaya di sini?",
            "Apakah lampu menyala?",
            "Cahaya di sini bagaimana?",
            "Bagaimana situasi pencahayaan?",
            "Apa status terbaru tentang keadaan?",
            "Bagaimana kondisi sekarang?",
            "Apa yang terdeteksi di sini?",
            "Apakah ada bahaya yang perlu diwaspadai?",
            "Ada indikasi asap atau gelap?",
            "Adakah perubahan pada suhu, kelembapan, atau cahaya?",
            "Apa kondisi suhu di sini?",
            "Apakah suhu terlalu panas atau dingin?",
            "Bagaimana kenyamanan suhu sekarang?",
            "Bagaimana kelembapan di ruangan?",
            "Apakah kelembapan tinggi?"
        ]
        selected_question = st.selectbox("Pilih pertanyaan:", ["Pilih pertanyaan..."] + questions, key="chatbot_select")
        if selected_question != "Pilih pertanyaan...":
            response = generative_chatbot_response(
                selected_question,
                mq2_value_latest,
                lux_value_latest,
                temperature_value_latest,
                humidity_value_latest
            )
            st.write(f"🤖: {response}")

    st.markdown("---")
    st.subheader("🔍 Prediksi Risiko Merokok")
    if st.button("Analisis Risiko Merokok"):
        prediction = predict_smoking_risk_rule_based(mq2_value_latest, lux_value_latest)
        st.write(prediction)
        if "Tinggi" in prediction:
            import asyncio
            asyncio.run(send_telegram_message(f"🚨 *Peringatan Risiko*: {prediction}"))

    st.markdown('</div>', unsafe_allow_html=True)

# --- ESP32-CAM TAB ---
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.header("Deteksi Merokok dengan ESP32-CAM")
    st.write("Mendeteksi aktivitas merokok secara real-time menggunakan kamera ESP32-CAM dan model YOLOv5.")

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        start_cam = st.checkbox("Mulai Deteksi", value=False, key="cam_start")
    with col2:
        auto_refresh_cam = st.checkbox("Aktifkan Auto-Refresh Kamera", value=True, key="cam_refresh")

    if start_cam:
        st.session_state.cam_running = True
        if 'model_cam' not in st.session_state or st.session_state.model_cam is None:
            st.session_state.model_cam = load_yolo_model()  # Muat model saat deteksi dimulai
        if auto_refresh_cam:
            run_camera_detection(frame_placeholder, status_placeholder)
        elif st.session_state.last_frame is not None:
            frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
            status_placeholder.info("Auto-refresh kamera dimatikan. Menampilkan gambar terakhir.")
        else:
            status_placeholder.warning("Tidak ada gambar terakhir untuk ditampilkan.")
    else:
        st.session_state.cam_running = False
        if st.session_state.last_frame is not None:
            frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
            status_placeholder.info("Kamera dimatikan. Menampilkan gambar terakhir.")
        else:
            status_placeholder.info("Klik 'Mulai Deteksi' untuk memulai streaming dari kamera.")

    st.markdown('</div>', unsafe_allow_html=True)
