# ============================================================================
# IMAGE CAPTURE SCRIPT — Run on Nicla Vision via OpenMV IDE
# ============================================================================
# Use this to capture training images FROM the Nicla Vision camera.
# These images will bridge the domain gap between dataset and device.
#
# HOW TO USE:
#   1. Upload this script to Nicla Vision
#   2. In OpenMV IDE, you will see the camera feed in the frame buffer
#   3. Point at the object you want to capture
#   4. Press the USER button (or just let it auto-capture every 2 seconds)
#   5. Images are saved to: /captured/<class_name>/img_001.jpg
#   6. Connect Nicla Vision as USB drive to copy images to your PC
#   7. Add these images to your Google Drive dataset folder
#   8. Retrain the model with the combined dataset
#
# CAPTURE ~20-30 IMAGES PER CLASS with varying:
#   - Angles (straight, slight left/right, above/below)
#   - Distances (close, medium, far)
#   - Lighting (bright, dim, mixed)
#   - Backgrounds (different rooms, surfaces)
# ============================================================================

import sensor
import image
import time
import os

# ============================================================================
# CONFIGURATION — Change CLASS_NAME before capturing each class!
# ============================================================================

# >>> CHANGE THIS for each object class you capture <<<
CLASS_NAME = "bag"

# All 12 classes — capture images for each one:
# "bag", "book", "bottle", "clear_path", "doorwindow", "dustbin",
# "human", "lift", "obstacle", "shoes", "stairs", "table&chair"

CAPTURE_INTERVAL_MS = 2000   # Auto-capture every 2 seconds
MAX_IMAGES = 30              # Stop after this many captures per class

# ============================================================================
# SETUP
# ============================================================================

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)   # 160x120 — matches deployment
sensor.set_vflip(False)
sensor.set_hmirror(False)
sensor.skip_frames(time=2000)

# Create output directories
SAVE_DIR = "/captured/" + CLASS_NAME
try:
    os.mkdir("/captured")
except:
    pass
try:
    os.mkdir(SAVE_DIR)
except:
    pass

# Count existing images
existing = 0
try:
    existing = len(os.listdir(SAVE_DIR))
except:
    pass

print("=" * 50)
print("  IMAGE CAPTURE for class: %s" % CLASS_NAME)
print("  Save directory: %s" % SAVE_DIR)
print("  Existing images: %d" % existing)
print("  Will capture up to %d images" % MAX_IMAGES)
print("  Auto-capture every %d ms" % CAPTURE_INTERVAL_MS)
print("=" * 50)
print("\nPoint camera at [%s] objects..." % CLASS_NAME)
print("Capturing starts NOW!\n")

# ============================================================================
# CAPTURE LOOP
# ============================================================================

count = existing
last_capture = time.ticks_ms()

while count < MAX_IMAGES + existing:
    img = sensor.snapshot()

    now = time.ticks_ms()
    if time.ticks_diff(now, last_capture) >= CAPTURE_INTERVAL_MS:
        count += 1
        filename = "%s/img_%03d.jpg" % (SAVE_DIR, count)
        img.save(filename, quality=90)
        print("[%d/%d] Saved: %s" % (
            count - existing, MAX_IMAGES, filename))
        last_capture = now

    # Draw capture info on screen (visible in OpenMV IDE)
    img.draw_string(2, 2, "Class: %s" % CLASS_NAME,
                    color=(255, 255, 255), scale=1)
    img.draw_string(2, 15, "Captured: %d/%d" % (count - existing, MAX_IMAGES),
                    color=(0, 255, 0), scale=1)

print("\n" + "=" * 50)
print("  DONE! Captured %d images for [%s]" % (MAX_IMAGES, CLASS_NAME))
print("  Files saved to: %s" % SAVE_DIR)
print("=" * 50)
print("\nNext steps:")
print("1. Change CLASS_NAME to the next class")
print("2. Run again to capture that class")
print("3. When done with all classes, connect Nicla as USB drive")
print("4. Copy /captured/ folder to your PC")
print("5. Add images to Google Drive dataset and retrain!")
