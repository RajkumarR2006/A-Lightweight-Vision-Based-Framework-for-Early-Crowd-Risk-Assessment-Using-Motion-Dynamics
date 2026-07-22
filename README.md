# A Lightweight Vision-Based Framework for Early Crowd Risk Assessment Using Motion Dynamics

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg">
  <img src="https://img.shields.io/badge/Status-Under%20Development-orange.svg">
  <img src="https://img.shields.io/badge/License-MIT-red.svg">
</p>

## 📌 Overview

Crowd-related incidents such as congestion, panic, and stampedes pose serious safety risks in public spaces including railway stations, airports, stadiums, shopping malls, and religious gatherings.

This project proposes a lightweight computer vision framework that continuously analyzes CCTV video streams to estimate crowd risk using motion dynamics and congestion analysis.

Unlike deep learning approaches requiring large computational resources, this framework is based primarily on classical computer vision techniques implemented using OpenCV, making it suitable for real-time deployment on resource-constrained systems.

---

# 🎯 Objectives

- Develop a lightweight vision-based crowd monitoring framework.
- Estimate crowd movement using Optical Flow.
- Detect moving pedestrians using Background Subtraction.
- Analyze crowd dynamics using handcrafted motion features.
- Compute a Crowd Risk Index (CRI).
- Classify crowd situations into:
  - 🟢 Safe
  - 🟡 Warning
  - 🔴 Critical

---

# 🚀 Proposed Pipeline

```
Input Video
      │
      ▼
Frame Extraction
      │
      ▼
Preprocessing
(Gray + Gaussian Blur)
      │
      ▼
Background Subtraction
      │
      ▼
Dense Optical Flow
      │
      ▼
Feature Extraction
 ┌──────────────────────────────┐
 │ Motion Instability Index     │
 │ Direction Conflict Index     │
 │ Density Estimation           │
 │ Congestion Pressure Index    │
 └──────────────────────────────┘
      │
      ▼
Crowd Risk Index (CRI)
      │
      ▼
Safe / Warning / Critical
```

---

# 💡 Motivation

Existing crowd monitoring systems generally detect anomalies only after abnormal behavior becomes evident or rely on computationally expensive deep learning models.

This project focuses on developing an interpretable, lightweight, and vision-only framework capable of continuously estimating crowd risk using traditional computer vision techniques.

---

# ✨ Features

- Image sequence processing
- Video frame extraction
- Image preprocessing
- Background subtraction (MOG2)
- Dense Optical Flow (Farneback)
- Motion feature extraction
- Crowd density estimation
- Congestion analysis
- Crowd Risk Index (CRI)
- Real-time visualization

---

# 🛠 Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

Future Integration

- Scikit-Learn
- SciPy
- Pandas

---

# 📂 Project Structure

```
OPEN_CV
│
├── datasets/
│
├── notebooks/
│
├── outputs/
│
├── src/
│   │
│   ├── features/
│   │      motion_instability.py
│   │      direction_conflict.py
│   │      density.py
│   │      congestion_pressure.py
│   │      crowd_risk.py
│   │
│   ├── background_subtraction.py
│   ├── config.py
│   ├── dataset_loader.py
│   ├── optical_flow.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── main.py
│
├── README.md
├── requirements.txt
└── roadmap.txt
```

---

# 📊 Dataset

This project uses publicly available crowd datasets.

- UCSD Pedestrian Dataset
- Crowd-11 Dataset

Future Evaluation

- ShanghaiTech Dataset

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/A-Lightweight-Vision-Based-Framework-for-Early-Crowd-Risk-Assessment-Using-Motion-Dynamics.git
```

Navigate into the project

```bash
cd A-Lightweight-Vision-Based-Framework-for-Early-Crowd-Risk-Assessment-Using-Motion-Dynamics
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python src/main.py
```

---

# 📈 Development Roadmap

- [x] Project Setup
- [x] Dataset Loading
- [x] Image Preprocessing
- [x] Background Subtraction
- [x] Dense Optical Flow
- [ ] Motion Instability Index
- [ ] Direction Conflict Index
- [ ] Density Estimation
- [ ] Congestion Pressure Index
- [ ] Crowd Risk Index
- [ ] Real-time Visualization
- [ ] Performance Evaluation
- [ ] Research Paper

---

# 🔬 Research Contribution

This project proposes a lightweight crowd risk assessment framework based on handcrafted motion dynamics instead of computationally intensive deep learning methods.

The framework introduces interpretable motion-based indicators to estimate crowd risk using only surveillance video streams.

---

# 📚 References

- UCSD Anomaly Detection Dataset
- Crowd-11 Dataset
- OpenCV Documentation

---

# 👨‍💻 Author

**Raj Kumar R**

B.Tech Artificial Intelligence & Data Science

Amrita Vishwa Vidyapeetham

---

## ⭐ Project Status

🚧 Active Development

This repository is being developed as part of an academic research project focusing on lightweight computer vision methods for crowd risk assessment.