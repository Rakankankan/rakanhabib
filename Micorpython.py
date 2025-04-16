import machine
import time
import ssd1306
import dht
import network
import ujson
import urequests  # pastikan urequests.py sudah di-upload ke board

# === KONFIGURASI WiFi ===
SSID = "moonstar"
PASSWORD = "17072005"

# === KONFIGURASI UBIDOTS ===
UBIDOTS_TOKEN = "BBUS-JBKLQqTfq2CPXNytxeUfSaTjekeL1K"  # Ganti dengan token milikmu
UBIDOTS_DEVICE_LABEL = "hsc345"
UBIDOTS_URL = "http://industrial.api.ubidots.com/api/v1.6/devices/{}".format(UBIDOTS_DEVICE_LABEL)

headers = {
    "X-Auth-Token": UBIDOTS_TOKEN,
    "Content-Type": "application/json"
}

# === KONFIGURASI PIN ===
DHT_PIN = 13
LDR_PIN = 34
PIR_PIN = 12
I2C_SDA = 21
I2C_SCL = 22
RED_LED_PIN = 14
YELLOW_LED_PIN = 15
GREEN_LED_PIN = 4
MQ2_PIN = 35

# === INISIALISASI I2C & OLED ===
i2c = machine.I2C(0, scl=machine.Pin(I2C_SCL), sda=machine.Pin(I2C_SDA))
oled = ssd1306.SSD1306_I2C(128, 64, i2c)

# === INISIALISASI PIN OUTPUT ===
red_led = machine.Pin(RED_LED_PIN, machine.Pin.OUT)
yellow_led = machine.Pin(YELLOW_LED_PIN, machine.Pin.OUT)
green_led = machine.Pin(GREEN_LED_PIN, machine.Pin.OUT)

# === INISIALISASI SENSOR ===
sensor = dht.DHT11(machine.Pin(DHT_PIN))

ldr_sensor = machine.ADC(machine.Pin(LDR_PIN))
ldr_sensor.width(machine.ADC.WIDTH_12BIT)
ldr_sensor.atten(machine.ADC.ATTN_11DB)

mq2_sensor = machine.ADC(machine.Pin(MQ2_PIN))
mq2_sensor.width(machine.ADC.WIDTH_12BIT)
mq2_sensor.atten(machine.ADC.ATTN_11DB)

# === KONEKSI KE WiFi ===
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)
    while not wlan.isconnected():
        time.sleep(1)
    print("Terhubung ke WiFi!")

# === BACA SENSOR ===
def read_dht11():
    try:
        sensor.measure()
        return sensor.humidity(), sensor.temperature()
    except OSError:
        return None, None

def read_ldr():
    return ldr_sensor.read()

def read_mq2():
    return mq2_sensor.read()

# === KIRIM KE UBIDOTS ===
def send_to_ubidots(temperature, humidity, lux, mq2_value):
    try:
        payload = ujson.dumps({
            "temperature": temperature,
            "humidity": humidity,
            "lux": lux,
            "mq2": mq2_value
        })
        response = urequests.post(UBIDOTS_URL, headers=headers, data=payload)
        print("Data terkirim ke Ubidots:", response.text)
        response.close()
    except Exception as e:
        print("Gagal kirim ke Ubidots:", e)

# === KONTROL LED ===
def control_alerts(temperature, mq2_value):
    global smoke_alert_active

    if temperature < 29:
        green_led.on()
        yellow_led.off()
        red_led.off()
    elif 29 <= temperature <= 30:
        green_led.off()
        yellow_led.on()
        red_led.off()
    else:
        green_led.off()
        yellow_led.off()
        red_led.on()

    if mq2_value > 600 and not smoke_alert_active:
        red_led.on()
        green_led.off()
        yellow_led.off()
        smoke_alert_active = True
        time.sleep(0.5)

    if mq2_value <= 600:
        smoke_alert_active = False

# === TAMPILKAN DI OLED ===
def show_main_data(humidity, temperature, lux, mq2_value):
    oled.fill(0)
    if humidity is not None and temperature is not None:
        oled.text('Suhu: {:.1f} C'.format(temperature), 0, 0)
        oled.text('Kelembapan: {:.1f} %'.format(humidity), 0, 10)
        oled.text('Lux: {}'.format(lux), 0, 20)
        oled.text('MQ2: {}'.format(mq2_value), 0, 30)
    else:
        oled.text('Sensor error!', 0, 0)
    oled.show()

# === MAIN ===
connect_wifi()

smoke_alert_active = False

while True:
    humidity, temperature = read_dht11()
    lux = read_ldr()
    mq2_value = read_mq2()

    if temperature is not None:
        control_alerts(temperature, mq2_value)
        send_to_ubidots(temperature, humidity, lux, mq2_value)

    show_main_data(humidity, temperature, lux, mq2_value)
    time.sleep(2)

