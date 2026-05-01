'''before running the code, click on the "new class folder" icon. Give it your class name. Then single click on the class created and run the code. Then click on "capture data" to click the images'''

import sensor
import time

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

clock = time.clock()

while True:
    clock.tick()
    img = sensor.snapshot()
    print(clock.fps())
