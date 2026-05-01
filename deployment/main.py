# ============================================================================
# AI-Powered Blind Assistance — Nicla Vision (Matching Edge Impulse)
# ============================================================================
# Camera: QVGA + 240x240 windowing (EXACT Edge Impulse setup)
# Model: INT8 TFLite, 6 classes
# ============================================================================

import sensor, image, time, ml, uos, gc, os
from machine import Pin, LED
import bluetooth
from micropython import const

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "blind_assist_int8.tflite"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.75
DETECTION_INTERVAL_MS = 500
SMOOTHING_WINDOW = 3

DANGER_CLASSES = {"human", "obstacle", "stairs"}
SAFE_CLASSES = {"clear_path"}

# ============================================================================
# BLE SETUP
# ============================================================================
_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)

ble = bluetooth.BLE()
ble.active(True)

UART_UUID = bluetooth.UUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
TX_UUID = bluetooth.UUID("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
_UART_SERVICE = (UART_UUID, ((TX_UUID, bluetooth.FLAG_NOTIFY | bluetooth.FLAG_READ),),)
((tx,),) = ble.gatts_register_services((_UART_SERVICE,))

ble_connected = False
ble_conn_handle = None

def bt_irq(event, data):
    global ble_connected, ble_conn_handle
    if event == _IRQ_CENTRAL_CONNECT:
        ble_conn_handle, _, _ = data
        ble_connected = True
        print("\n[BLE] Phone Connected!")
    elif event == _IRQ_CENTRAL_DISCONNECT:
        ble_conn_handle = None
        ble_connected = False
        print("\n[BLE] Disconnected. Advertising...")
        advertise()

ble.irq(bt_irq)

def advertise():
    name = "Nicla Vision AI"
    payload = bytearray((2, 1, 6, len(name) + 1, 9)) + name.encode('utf-8')
    ble.gap_advertise(100000, payload)

print("[OK] BLE Initialized (will advertise after camera/model setup)")

# ============================================================================
# HARDWARE
# ============================================================================
red_led = LED("LED_RED")
green_led = LED("LED_GREEN")
blue_led = LED("LED_BLUE")

try:
    buzzer = Pin("PB3", Pin.OUT_PP)
    HAS_BUZZER = True
except:
    HAS_BUZZER = False
    print("[INFO] No buzzer — LED-only mode")

# ============================================================================
# LOAD MODEL & LABELS
# ============================================================================
labels = []
try:
    with open(LABELS_PATH, "r") as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)
    print("[OK] Loaded %d classes: %s" % (len(labels), str(labels)))
except:
    labels = ["clear_path", "door", "human", "object", "obstacle", "stairs"]
    print("[WARN] Using default 6-class labels")

NUM_CLASSES = len(labels)

# ============================================================================
# CAMERA — MATCH TRAINING PREPROCESSING EXACTLY
# ============================================================================
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)      # 320x240
# DO NOT use set_windowing! Training used squashed full frames.
sensor.skip_frames(time=2000)

print("[OK] Camera: QVGA (will squash to 96x96 to match training)")

# ============================================================================
# LOAD MODEL (Edge Impulse style memory management)
# ============================================================================
print("Loading model...")
net = ml.Model(MODEL_PATH, load_to_fb=True)
print("[OK] Model loaded! Input: %s Output: %s" % (str(net.input_shape), str(net.output_shape)))

# Start BLE advertising AFTER camera and model are ready
advertise()
print("[OK] BLE now advertising as 'Nicla Vision AI'")

# ============================================================================
# FEEDBACK
# ============================================================================
def beep(duration_ms, count=1, gap_ms=100):
    if not HAS_BUZZER: return
    for i in range(count):
        buzzer.high()
        time.sleep_ms(duration_ms)
        buzzer.low()
        if i < count - 1: time.sleep_ms(gap_ms)

def set_leds(red=False, green=False, blue=False):
    if red: red_led.on()
    else: red_led.off()
    if green: green_led.on()
    else: green_led.off()
    if blue: blue_led.on()
    else: blue_led.off()

def feedback_for_class(class_name, confidence):
    if class_name in SAFE_CLASSES:
        set_leds(green=True)
        beep(50, count=2, gap_ms=200)
        return "SAFE"
    elif class_name in DANGER_CLASSES:
        set_leds(red=True)
        beep(100, count=3, gap_ms=50)
        return "DANGER"
    else:
        set_leds(blue=True)
        beep(150, count=1)
        return "OBJECT"

# ============================================================================
# SMOOTHING
# ============================================================================
prediction_buffer = []

def smooth_prediction(new_pred, new_conf):
    global prediction_buffer
    prediction_buffer.append((new_pred, new_conf))
    if len(prediction_buffer) > SMOOTHING_WINDOW:
        prediction_buffer.pop(0)
    pred_counts = {}
    for pred, conf in prediction_buffer:
        if pred not in pred_counts:
            pred_counts[pred] = {"count": 0, "total_conf": 0.0}
        pred_counts[pred]["count"] += 1
        pred_counts[pred]["total_conf"] += conf
    best_pred = max(pred_counts.keys(), key=lambda k: pred_counts[k]["count"])
    avg_conf = pred_counts[best_pred]["total_conf"] / pred_counts[best_pred]["count"]
    return best_pred, avg_conf

# ============================================================================
# MAIN LOOP
# ============================================================================
print("\n" + "=" * 50)
print("  Blind Assistance — 6 Class Detection")
print("=" * 50 + "\n")

prev_class = ""
frame_count = 0
total_inference_ms = 0
fps_start_time = time.ticks_ms()
last_detection_ms = 0

try:
    while True:
        # 1. Take a snapshot continuously to keep the IDE video feed smooth
        img = sensor.snapshot()
        
        current_ms = time.ticks_ms()
        
        # 2. Only run inference every DETECTION_INTERVAL_MS
        if time.ticks_diff(current_ms, last_detection_ms) >= DETECTION_INTERVAL_MS:
            start_ms = time.ticks_ms()
            last_detection_ms = current_ms

            # CRITICAL FIX: Manually squash 320x240 to 96x96 (aspect ratio 4:3 -> 1:1)
            img_squashed = img.copy(x_scale=0.3, y_scale=0.4)
            raw_output = net.predict([img_squashed])[0].flatten().tolist()

            inference_ms = time.ticks_diff(time.ticks_ms(), start_ms)
            total_inference_ms += inference_ms
            frame_count += 1

            max_val = max(raw_output)
            max_idx = raw_output.index(max_val)
            confidence = max_val
            class_name = labels[max_idx] if max_idx < len(labels) else "unknown"

            smoothed_class, smoothed_conf = smooth_prediction(class_name, confidence)

            if smoothed_conf >= CONFIDENCE_THRESHOLD:
                if smoothed_class != prev_class:
                    status = feedback_for_class(smoothed_class, smoothed_conf)
                    prev_class = smoothed_class
                    print("[%s] %s (%.1f%%) | %dms" % (
                        status, smoothed_class.upper(), smoothed_conf * 100, inference_ms))

                    if ble_connected and ble_conn_handle is not None:
                        try:
                            msg = "Detected: " + smoothed_class + ","
                            ble.gatts_notify(ble_conn_handle, tx, msg.encode('utf-8'))
                            print("      -> Sent to phone")
                        except Exception as e:
                            print("      -> BLE Error:", e)
            else:
                set_leds()
                prev_class = ""

            if frame_count % 30 == 0:
                elapsed = time.ticks_diff(time.ticks_ms(), fps_start_time)
                fps = 30000.0 / elapsed if elapsed > 0 else 0
                avg_ms = total_inference_ms / frame_count
                print("[FPS (Inference): %.1f | Avg: %.1fms]" % (fps, avg_ms))
                fps_start_time = time.ticks_ms()

except KeyboardInterrupt:
    set_leds()
    if HAS_BUZZER: buzzer.low()
    ble.active(False)
    print("\n[STOPPED]")
except Exception as e:
    set_leds(red=True)
    print("[ERROR]", e)
    raise e
