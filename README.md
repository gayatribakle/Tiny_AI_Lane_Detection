<h1>🛣️ Tiny-AI Lane Detection</h1>

A lightweight, fast, and accurate lane detection system designed for mobile and edge devices. This project uses Tiny AI models (quantized + optimized) to detect road lanes in real time with minimal compute and power usage.

---

**🚀 Features**

🔹 Real-time lane detection on CPU, mobile, or low-end GPUs

🔹 Tiny, optimized model (< 5–10 MB depending on version)

🔹 ONNX / TFLite support

🔹 Works on live camera feed, videos, and images

🔹 Fast inference (20–60 FPS depending on device)

🔹 Ideal for ADAS, autonomous vehicles, robotics, and IoT

---

**📁 Project Structure**
```
tiny-lane-unet/
├── scripts/                 # training, inference, generation
├── models_def/              # tiny UNet model
├── utils/                   # dataset loaders
├── data/                    # synthetic dataset
└── models/                  # saved models

📁 Output folders created automatically

data/
 └── lanes_synthetic/
      ├── images/
      └── masks/

tiny-lane-unet/
│
├── scripts/
│     ├── train_lane.py
│     ├── gen_lanes.py
│
├── models_def/
│     ├── __init__.py
│     └── tiny_unet.py


```

---
**Demo**

🧪 Results

| Metric            | Value                           |
| ----------------- | ------------------------------- |
| Model Size        | ~5–10 MB                        |
| FPS (CPU)         | 20–35 FPS                       |
| FPS (GPU/Android) | 30–60 FPS                       |
| Accuracy          | ~92–95% lane detection accuracy |

---

**🌍 Necessity of Tiny AI Lane Detection in the Real World**

Lane detection plays a crucial role in modern transportation systems, especially as the world moves toward smarter and safer mobility. Traditional lane detection methods often require heavy computational power and high-end hardware, which makes them difficult to deploy on real vehicles, low-cost devices, or mobile platforms. A Tiny AI lane detection system solves this problem by offering fast and accurate lane recognition using lightweight models that can run efficiently on edge devices, smartphones, low-power CPUs, and embedded systems. This is essential for Advanced Driver Assistance Systems (ADAS), where real-time lane information helps prevent accidents caused by lane drifting, driver drowsiness, and poor visibility. It also supports the development of autonomous vehicles, delivery robots, and smart traffic systems by enabling reliable navigation without depending on expensive hardware. The compact and optimized nature of Tiny AI models reduces power consumption, latency, and cost, making lane detection accessible to developing countries, low-budget projects, and IoT applications. Overall, this project is necessary because it provides a scalable, affordable, and energy-efficient solution to improve road safety, enhance driving comfort, and accelerate the adoption of intelligent transportation technologies.

---
