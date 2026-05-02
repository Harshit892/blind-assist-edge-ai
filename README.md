# WNAVI- Wearable Navigation Aid Utilizing Lightweight CNN for Visually Impaired

> **Edge AI Course Project (CP330)** — Real-time object classification for visually impaired navigation using Arduino Nicla Vision.
> **Course Link** — https://www.samy101.com/edge-ai-26/ 

## ✨ Highlights

- **True on-device Edge AI** — All inference runs locally on the Arduino Nicla Vision with no cloud dependency.
- **73x model compression** — From 8.8 MB MobileNetV2 teacher to 120 KB INT8 TFLite via Knowledge Distillation + Pruning + QAT.
- **5-class real-time classification** — clear_path, human, door, obstacle, stairs at <200ms per frame.
- **Multi-modal feedback** — RGB LEDs (Green/Red/Blue) + buzzer alerts + BLE notifications to smartphone.
- **End-to-end pipeline** — Data collection → CNN → Transfer Learning → KD → Pruning → QAT → INT8 → Deploy.

## 📁 Repository Structure

```
.
├── dataset/                    # Full image dataset (12 original classes)
│   ├── bag/
│   ├── book/
│   ├── bottle/
│   ├── clear_path/
│   ├── doorwindow/
│   ├── dustbin/
│   ├── human/
│   ├── lift/
│   ├── obstacle/
│   ├── shoes/
│   ├── stairs/
│   └── table&chair/
├── training/
│   └── EdgeAIproject.ipynb             # Complete training notebook (Colab)
├── deployment/
│   ├── main.py                 # OpenMV deployment script for Nicla Vision
│   ├── blind_assist_int8.tflite # Final INT8 quantized model
│   └── labels.txt              # Class labels file
├── data-collection/
│   └── capture_images.py       # On-device image capture script
├── images/                     # Result screenshots and diagrams
├── report.md                   # Detailed project report
└── README.md                   # This file
```

## 🛠️ Hardware Required

| Component | Purpose |
|-----------|---------|
| **Arduino Nicla Vision** | MCU with camera, BLE (STM32H747, 1MB RAM) |
| **USB Power Bank** | Portable power for helmet-mounted operation |
| **Buzzer** | Acoustic danger/safety alerts |
| **Safety Helmet** | Mounting platform for the device |
| **Smartphone** | BLE companion app (MIT App Inventor) |

## 🚀 Quick Start — Reproduce This Project

### Step 1: Train the Model (Google Colab)

1. Open `training/EdgeAIproject.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload the `dataset/` folder to your Google Drive.
3. Update the `DATASET_PATH` variable in the notebook to point to your dataset.
4. Run all cells. The notebook will:
   - Train a Custom CNN and MobileNetV2 Teacher model.
   - Perform Knowledge Distillation to create a tiny Student model.
   - Apply iterative magnitude-based pruning (30% → 70% sparsity).
   - Apply Quantization Aware Training (QAT).
   - Export `blind_assist_int8.tflite` and `labels.txt`.
5. Download the generated `.tflite` and `labels.txt` files.

### Step 2: Deploy to Arduino Nicla Vision

1. Install [OpenMV IDE](https://openmv.io/pages/download).
2. Connect the Nicla Vision via USB.
3. Copy these files to the Nicla Vision's internal storage:
   - `deployment/blind_assist_int8.tflite`
   - `deployment/labels.txt`
   - `deployment/main.py`
4. Open `main.py` in OpenMV IDE and click **Run**.

### Step 3: Smartphone App (Optional)

1. Open [MIT App Inventor](https://appinventor.mit.edu/).
2. Import the companion `.aia` file (if provided) or create a BLE UART receiver app.
3. Connect to "Nicla Vision AI" via BLE.
4. The flow diagram of MIT app inventor is given on `images/appflow`

## 📊 Results Summary

| Model Stage | Technique | Accuracy | Size |
|-------------|-----------|----------|------|
| Decision Tree | Baseline | ~76.68% | N/A |
| Custom CNN | From scratch | ~95.69% | 140 KB |
| MobileNetV2 (Teacher) | Transfer Learning | ~99.20% | 8.8 MB |
| KD Student | Knowledge Distillation | ~93.77% | 80 KB |
| Pruned Student | Magnitude Pruning | ~93.77% | 60 KB |
| After QAT | Quantization Aware | ~91.37% | 60 KB |
| **Final INT8 TFLite** | **Full pipeline** | **~91.21%** | **118 KB** |

## 📄 Full Report

See [report.md](report.md) for the complete project report with methodology, algorithms, challenges, and analysis.

## 🔗 References

- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
- [OpenMV Documentation](https://docs.openmv.io/)
- [Arduino Nicla Vision](https://docs.arduino.cc/hardware/nicla-vision/)
- [TF Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [Edge Impulse](https://www.edgeimpulse.com/)
