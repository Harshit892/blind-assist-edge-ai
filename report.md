# WNAVI: Wearable Navigation Aid for Visually Impaired — Project Report

**Course:** CP330 — Edge AI by Prof. Pandarasamy Arjunan, IISc Bangalore 

**Team Members:** Harshit Garhewal (25937), Anshul Verma (25881), Sourin Das (26033), Tushar Dewangan (26361)  

**Deployment Target:** Arduino Nicla Vision  

**Date:** 1, May 2026

---

## 1. Introduction

### 1.1 Problem Statement

Approximately 285 million people worldwide are visually impaired (WHO). Navigating complex indoor and outdoor environments—corridors, staircases, crowded spaces—is dangerous without real-time awareness of surrounding objects. Existing assistive technologies such as white canes and guide dogs provide limited contextual information and cannot identify specific object categories.

### 1.2 Proposed Solution

We propose an AI-powered wearable device that provides real-time object classification and multi-modal feedback to visually impaired users. The system uses a helmet-mounted Arduino Nicla Vision with an onboard camera to classify the scene into 5 semantic categories and provide instant audio (buzzer), visual (LED), and mobile (BLE) alerts.

### 1.3 Why Edge AI?

| Requirement | Edge AI Advantage |
|------------|------------------|
| **Privacy** | Images never leave the device |
| **Latency** | <200ms inference, no network delay |
| **Availability** | No internet/cloud dependency |
| **Power** | Runs on battery for portable use |

---

## 2. Hardware Platform

### 2.1 Arduino Nicla Vision

| Feature | Specification |
|---------|--------------|
| MCU | STM32H747 Dual-core (Cortex-M7 @ 480MHz + Cortex-M4 @ 240MHz) |
| RAM | 1 MB |
| Flash | 2 MB internal + 16 MB external QSPI |
| Camera | 2MP GC2145 color camera |
| Connectivity | WiFi + Bluetooth Low Energy (BLE) |
| Sensors | 6-axis IMU, microphone |
| Size | 22.86 × 22.86 mm |

The Nicla Vision was chosen for its compact form factor, built-in camera, and native TensorFlow Lite Micro support via OpenMV firmware—making it ideal for helmet-mounted wearable deployment.

<!-- INSERT: Photo of hardware setup / helmet mount -->

---

## 3. System Architecture

```
┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
│  Camera  │───>│ Preprocessing │───>│  TFLite INT8 │───>│   Decision   │
│ 320×240  │    │ Squash 96×96  │    │    Model     │    │    Logic     │
└──────────┘    └───────────────┘    └──────────────┘    └──────┬───────┘
                                                                │
                                         ┌──────────────────────┼──────────────────┐
                                         ▼                      ▼                  ▼
                                  ┌─────────────┐     ┌──────────────┐    ┌──────────────┐
                                  │  LED Alert   │     │    Buzzer    │    │  BLE → Phone │
                                  │ G / R / B    │     │   Patterns   │    │   via UART   │
                                  └─────────────┘     └──────────────┘    └──────────────┘
```

**Key Design Decisions:**
1. **Image squashing** (not cropping) to preserve the full field of view and match training preprocessing.
2. **Non-blocking inference loop** — `sensor.snapshot()` runs continuously for smooth camera feed; inference triggers every 500ms via timer.
3. **3-frame prediction smoothing** — Rolling buffer eliminates flickering between classes.
4. **BLE UART** — Notifies a paired MIT App Inventor smartphone app.

<!-- INSERT: System architecture diagram -->

---

## 4. Dataset

### 4.1 Data Collection

Images were captured directly on the Nicla Vision using a custom MicroPython script (`capture_images.py`). A push button triggers image capture, storing photos to the device's internal storage. Images were collected across multiple environments (classrooms, corridors, labs, outdoor walkways) to maximize diversity.

### 4.2 Original Classes (12)

The raw dataset contains 12 classes with ~4,625 total images:

| Class |
| book |
| lift |
| bottle |
| obstacle |
| clear_path |
| shoes |
| doorwindow |
| stairs |
| dustbin |
|table&chair |

### 4.3 Merged Classes (5)

To reduce inter-class confusion and improve deployment accuracy, we merged semantically similar classes:

| Merged Class | Original Classes | Category |
|-------------|-----------------|----------|
| **clear_path** | clear_path | SAFE |
| **human** | human | DANGER |
| **door** | doorwindow | OBJECT |
| **obstacle** | obstacle, table&chair, dustbin, bag, book, bottle, shoes | DANGER |
| **stairs** | stairs, lift | DANGER |

### 4.4 Data Augmentation

```python
ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=25.0,
    fill_mode='nearest',
    validation_split=0.15
)
```

These augmentations simulate real-world helmet camera variations: head tilts (rotation), walking movement (shifts), zoom variation (distance changes), and lighting changes (brightness/channel shift).

### 4.5 Data Split

| Split | Samples |
|-------|---------|
| Training | 3,342 (80%) |
| Validation | 587 (15%) |
| Test | 696 (5% held out) |

<!-- INSERT: Data augmentation examples screenshot -->
<!-- INSERT: Dataset distribution bar chart -->

---

## 5. Model Development Pipeline

```
Step 1: Data Collection & Augmentation                 
    ↓
Step 2: Baseline — Decision Tree                       
    ↓
Step 3: Custom CNN — From Scratch                     
    ↓
Step 4: Teacher — MobileNetV2 Transfer Learning        
    ↓
Step 5: Model Efficiency Metrics Analysis              
    ↓
Step 6: Knowledge Distillation (Teacher → Student)     
    ↓
Step 7: Iterative Magnitude-Based Pruning              
    ↓
Step 8: Quantization Aware Training (QAT)              
    ↓
Step 9: INT8 TFLite Conversion & Calibration           
    ↓
Step 10: Deploy to Arduino Nicla Vision                
```

---

## 6. Model Training & Results

### 6.1 Decision Tree Baseline

**Approach:** Flatten 96×96×3 images into 27,648-dimensional feature vectors and train a Decision Tree classifier.

```python
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=15)
dt_model.fit(X_train_flat, y_train)
```

**Purpose:** Establishes a non-deep-learning baseline. Demonstrates that raw pixel features are insufficient for complex scene classification, motivating the use of learned CNN features.

**Result:** ~86% accuracy — strong diagonal in the confusion matrix but significant misclassifications between visually similar classes (obstacle vs. clear_path).

<!-- INSERT: Decision Tree confusion matrix -->

### 6.2 Custom CNN

**Architecture:**
```
Input (96×96×3)
    ↓
Conv2D(16, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)    [48×48×16]
    ↓
Conv2D(32, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)    [24×24×32]
    ↓
Conv2D(64, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)    [12×12×64]
    ↓
GlobalAveragePooling2D                                       [64]
    ↓
Dense(64, ReLU) → Dropout(0.3) → Dense(5, softmax)
```

**Design Decisions:**
- `GlobalAveragePooling2D` instead of `Flatten` reduces parameters from 36,864 to 64.
- `BatchNormalization` enables faster convergence with higher learning rates.
- Small filter progression (16→32→64) keeps total parameters at ~35K.

**Training:** 25 epochs, Adam optimizer (lr=1e-3), EarlyStopping (patience=5), ReduceLROnPlateau.

**Result:** ~98% validation accuracy. Near-perfect confusion matrix.

<!-- INSERT: Custom CNN training curves (loss + accuracy) -->
<!-- INSERT: Custom CNN confusion matrix -->

### 6.3 Teacher Model — MobileNetV2

**Transfer Learning Strategy:**
1. **Stage 1 (Feature Extraction):** Freeze entire MobileNetV2 base, train only the classification head for 15 epochs.
2. **Stage 2 (Fine-Tuning):** Unfreeze the last 30 layers, train end-to-end with a lower learning rate (1e-5) for 25 epochs.

**Architecture:**
```
MobileNetV2 (pretrained on 1.4M ImageNet images)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, ReLU) → Dropout(0.3) → Dense(5, softmax)
```

**Result:** ~99.5% validation accuracy. The Teacher achieves near-perfect classification and serves as the source of "dark knowledge" for distillation. However, at ~8.8 MB (Float32), it is far too large for the Nicla Vision's 2 MB flash.

<!-- INSERT: Teacher MobileNetV2 training curves -->
<!-- INSERT: Teacher MobileNetV2 confusion matrix -->

---

## 7. Model Compression Techniques

### 7.1 Knowledge Distillation

**Concept:** Transfer "dark knowledge" from the large Teacher model to a tiny Student model. The Teacher's soft probability outputs reveal inter-class relationships (e.g., "human" is more similar to "obstacle" than to "clear_path") that hard one-hot labels cannot convey.

**Temperature Scaling:**
- At T=1: Hard probabilities — sharp peak at the true class.
- At T=5: Soft probabilities — probability mass spreads across similar classes, revealing learned structure.

**Distillation Loss:**
```
L = α · KL_Divergence(soft_student, soft_teacher) · T²
  + (1 - α) · CrossEntropy(student_pred, hard_labels)
```
- **α = 0.7** — 70% weight on distillation, 30% on hard labels.
- **T = 5** — Temperature for softening.
- **T² scaling** — Compensates for reduced gradient magnitudes at high temperature.

**Student Architecture:**
```
Conv2D(16, 3×3, same, ReLU) → MaxPool(2×2)
Conv2D(32, 3×3, same, ReLU) → MaxPool(2×2)
Flatten → Dropout(0.25) → Dense(5, softmax)
```
Target: <100K parameters (95% smaller than Teacher).

**Custom Training Loop:**
```python
# Offline: Generate soft targets from Teacher
teacher_preds = teacher_model.predict(X_train)
soft_targets = softmax(log(teacher_preds) / T)

# Online: Train Student with combined loss
for epoch in range(epochs):
    student_preds = student(x_batch)
    student_soft = softmax(log(student_preds) / T)
    loss = α * KL_Div(soft_targets, student_soft) * T²
         + (1-α) * CE(hard_labels, student_preds)
```

**Result:** KD Student achieves ~95% accuracy with <100K parameters — nearly matching the Teacher while being 95% smaller. The Vanilla Student (trained without KD) achieved significantly lower accuracy, proving distillation's value.

<!-- INSERT: KD vs Vanilla Student comparison chart -->

### 7.2 Magnitude-Based Pruning

**Concept:** Remove weights closest to zero — they contribute least to the model's output. Gradually increase sparsity during fine-tuning to allow the model to adapt.

**Iterative Pruning Schedule (3 Rounds):**

| Round | Initial Sparsity | Final Sparsity | Learning Rate |
|-------|-----------------|---------------|---------------|
| 1 | 20% | 40% | 1e-4 |
| 2 | 40% | 60% | 5e-5 |
| 3 | 60% | 70% | 3.3e-5 |

**Implementation:**
```python
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.20, final_sparsity=0.70,
        begin_step=0, end_step=end_step
    )
}
model_for_pruning = prune_low_magnitude(kd_student, **pruning_params)
```

**Callbacks:** `UpdatePruningStep()` updates the pruning mask at each training step. `PruningSummaries()` logs sparsity metrics.

**Result:** 70% of weights become zero with <2% accuracy drop (~93%). The zero-valued weights enable better compression when packaged with gzip/zip.

<!-- INSERT: Weight distribution before/after pruning -->

### 7.3 Quantization Aware Training (Lab 07/08)

**Problem:** Post-Training Quantization (converting Float32 to INT8 after training) can cause severe accuracy degradation, especially in small models.

**QAT Solution:** Insert "fake quantization" nodes during training that simulate INT8 precision constraints. The model learns to be robust to quantization noise while optimizing.

```python
quant_aware_model = tfmot.quantization.keras.quantize_model(pruned_model)
quant_aware_model.compile(optimizer=Adam(1e-4), ...)
quant_aware_model.fit(X_train, y_train, epochs=15, ...)
```

**Result:** QAT recovers ~1% accuracy compared to naive post-training quantization.

### 7.4 INT8 TFLite Conversion 

```python
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen  # 500 calibration samples
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
```

**Final model:** ~120 KB INT8 TFLite — a **73x compression** from the 8.8 MB Teacher.

---

## 8. Model Comparison & Results

| Stage | Technique | Lab | Accuracy | Size | Parameters |
|-------|-----------|-----|----------|------|-----------|
| Baseline | Decision Tree | 04 | ~86% | N/A | N/A |
| Custom CNN | From scratch | 05 | ~98% | 140 KB | ~35K |
| Teacher | MobileNetV2 | 05 | ~99.5% | 8.8 MB | ~2.3M |
| KD Student | Distillation | 10 | ~95% | 80 KB | <100K |
| Pruned | 70% sparsity | 09 | ~93% | 60 KB | <100K |
| After QAT | Quant-aware | 08 | ~94% | 60 KB | <100K |
| **INT8 TFLite** | **Full pipeline** | **07** | **~93%** | **120 KB** | **<100K** |

**Key Achievement:** 73x size reduction (8.8 MB → 120 KB) with <6% accuracy drop, fitting comfortably within the Nicla Vision's 1 MB RAM.

<!-- INSERT: Accuracy comparison bar chart -->
<!-- INSERT: Model size comparison bar chart -->
<!-- INSERT: Final INT8 confusion matrix -->

---

## 9. Deployment

### 9.1 Deployment Script (`main.py`)

The deployment script runs on the Nicla Vision via OpenMV's MicroPython firmware:

```python
# Camera setup — full QVGA frame
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320×240

# Load model into framebuffer
net = ml.Model("blind_assist_int8.tflite", load_to_fb=True)

# Main loop (non-blocking)
while True:
    img = sensor.snapshot()
    if ticks_diff(now, last) >= INTERVAL:
        img_sq = img.copy(x_scale=0.3, y_scale=0.4)  # Squash to 96×96
        output = net.predict([img_sq])[0].flatten().tolist()
        class_name = labels[argmax(output)]
        # Provide feedback...
```

### 9.2 Feedback System

| Detection | LED | Action |
|-----------|-----|--------|
| Clear Path | 🟢 Green | Safe to walk |
| Human / Obstacle / Stairs | 🔴 Red | DANGER — Stop! |
| Door | 🔵 Blue | Object nearby |
| Low confidence (<75%) | ⚫ Off | Ignore |

### 9.3 BLE Communication

A BLE UART service (`6E400001-B5A3-F393-E0A9-E50E24DCCA9E`) sends detection results to a paired smartphone running an MIT App Inventor companion app.

---

## 10. Deployment Challenges & Solutions

### Challenge 1: "Failed to Allocate Tensors"

**Problem:** The TFLite model crashed on load with `ValueError: Failed to allocate tensors`.

**Root Cause:** The MicroPython BLE stack was initialized before the model, consuming large amounts of heap memory and fragmenting it. The Conv2D first layer requires 96×96×16 = 144 KB for a single activation map. With the camera framebuffer (240×240×2 = 112 KB) also allocated, insufficient contiguous RAM remained for the tensor arena.

**Solution:**
- Reordered initialization: Load model into framebuffer **first**, then initialize BLE.
- Used `load_to_fb=True` to place model weights in the camera's framebuffer memory.
- Added `gc.collect()` before model loading.

### Challenge 2: Preprocessing Mismatch (Squash vs Crop)

**Problem:** The model achieved ~90% accuracy in Colab but misclassified almost everything on the device.

**Root Cause:** A critical domain discrepancy between training and deployment preprocessing:
- **Training (Keras):** `flow_from_directory` resized 320×240 images to 96×96 by **squashing** the 4:3 aspect ratio to 1:1.
- **Deployment (OpenMV):** `set_windowing((240, 240))` performed a **center crop**, cutting off 40px from left and right edges. Objects near the edges were completely invisible.

**Solution:**
- Removed `set_windowing()` — capture the full 320×240 frame.
- Manually squash: `img.copy(x_scale=0.3, y_scale=0.4)` → exact 96×96 matching Keras.

### Challenge 3: Shortcut Learning (Background Overfitting)

**Problem:** The custom CNN trained from scratch memorized background textures instead of object features. Example: all bottles were photographed on a brown desk → model learned "brown desk = bottle".

**Root Cause:** Small datasets (~3,000 images) with limited background diversity cause tiny CNNs to exploit spurious correlations.

**Solution:**
- **Transfer Learning:** MobileNetV1 (alpha=0.25) pre-trained on 1.4M ImageNet images.
- **Heavy augmentation:** rotation, brightness, zoom, channel shift.
- **Dataset expansion:** Captured additional images across diverse environments.

### Challenge 4: Camera Lag in OpenMV IDE

**Problem:** The camera feed displayed at 2 FPS — a choppy slideshow.

**Root Cause:** `time.sleep_ms(500)` at the end of the main loop blocked the entire camera refresh.

**Solution:** Replaced blocking sleep with a non-blocking `time.ticks_diff()` timer. Camera runs `sensor.snapshot()` at full speed (30+ FPS) while inference triggers only every 500ms.

---

## 11. Limitations
-Classification Only, No Localization — The model only tells what is in front (e.g., "obstacle") but not where it is (left, right, center) or how far it is. A blind user needs spatial awareness, not just object labels.

-Limited Class Coverage — Only 5 classes are supported (clear_path, human, door, obstacle, stairs). Real-world environments contain many more hazards — vehicles, potholes, wet floors, traffic signals, animals — that the system cannot detect.

-Lighting & Environment Sensitivity — The model was trained primarily in indoor/campus environments with controlled lighting. Performance degrades significantly in low-light conditions, nighttime, direct sunlight glare, or rainy/foggy weather.

-Single Frame Classification (No Temporal Context) — Each frame is classified independently. The system cannot track moving objects (e.g., a person walking toward the user) or understand motion context (e.g., distinguishing a parked car from an approaching one)

## 12. Future Work

- **Expanded Dataset:** 10,000+ images across more diverse environments.
- **Object Localization:** Detect WHERE objects are (left/right/center) using FOMO.
- **Distance Estimation:** Monocular depth cues from the camera.
- **Voice Feedback:** Replace buzzer with TTS via Bluetooth earpiece.
- **Federated Learning (Lab 12):** Collaborative model updates across devices.
- **GPS Integration:** Outdoor navigation assistance.

---

## 13. References
[1] C.-D. Sahoo, "Image-Classification-Under-256KB," GitHub, 2023. [Online]. Available: https://github.com/Chinmay-Deep-Sahoo/Image-Classification-Under-256KB. [Accessed: May 1, 2026].

[2] [Uploader Name], "[Video Title]," YouTube, [Year]. [Online]. Available: https://youtu.be/zeybEOM2BHY?si=vKVm2QDbGAw6GP3D. [Accessed: May 1, 2026].

[3] [Author Name], "[Document/File Title]," Google Share, [Year]. [Online]. Available: https://share.google/FC6kqlOlAXq5ktKyz. [Accessed: May 1, 2026].

[4] BrianMacG, "Arduino Deployment with Nicla Vision - Initial Success Followed by 'Failed to run classifier'," Edge Impulse Forum, Apr. 19, 2025. [Online]. Available: https://forum.edgeimpulse.com/t/arduino-deployment-with-nicla-vision-initial-success-followed-by-failed-to-run-classifier/13868. [Accessed: May 1, 2026].

[5] milnepe, "Image Recognition with Arduino Nicla Vision: A Radxa ROCK SBC Classifier," DesignSpark, Jun. 10, 2024. [Online]. Available: https://www.rs-online.com/designspark/image-recognition-with-arduino-nicla-vision-a-radxa-rock-sbc-classifier.

**Tools:** TensorFlow, TFLite, TF Model Optimization Toolkit, OpenMV IDE, MicroPython, Arduino Nicla Vision, Google Colab, MIT App Inventor.
