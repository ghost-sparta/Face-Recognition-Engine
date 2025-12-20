## Face Recognition Engine
Fast, Optimized, Production-Ready Face Encoding & Real-Time Recognition System

This repository provides a complete face recognition pipeline built with Python, OpenCV, and face_recognition.
It includes:

🔹 Face Enrollment System (via images or webcam)

🔹 Encoding Manager (saves/loads .pkl encoding datasets)

🔹 High-Performance Recognition Engine

🔹 Threaded, Scalable, Real-Time Processing

🔹 FPS-Optimized Detection with Caching & Frame Skipping

## Designed for real-world use in:
✔ CCTV Surveillance
✔ Smart Home Systems
✔ Access Control / Door Systems
✔ Security & Pentesting Tools
✔ Machine Vision Projects

## 📸 Features
## 🔐 Face Encoder & Enrollment

Encode multiple faces from a folder structure

Add new faces from webcam (press SPACE to capture)

Automatically updates and saves face encodings

Compatible with old & new encoding formats

## ⚡ High-Performance Recognition Engine

Multi-threaded processing

Frame skipping (process every Nth frame)

Cached detections for smoother FPS

Scaled-down processing for speed

Confidence scoring for all detected faces

Real-time FPS + system stats overlay

Robust logging & error handling

## 🧠 Robust Architecture

Handles corrupted files gracefully

Auto-creates missing directories

Backwards compatible with old pickle formats

Thread-safe cached results

Easily extendable (Flask, FastAPI, PyQt5, CCTV systems, etc.)

## 📁 Project Structure
```bash
project/
│
├── data/
│   └── face_encodings.pkl          # Auto-generated face dataset
│
├── known_faces/
│   ├── John/
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── Alice/
│       ├── img1.png
│       └── img2.jpg
│
├── face_recognition_module.py      # Production recognition engine
└── face encoder and enroller.py    # Encoder + enrollment tool
```

## 🔧 Installation
Install Python dependencies:
pip install opencv-python face_recognition numpy

Windows users (recommended):

Install dlib prebuilt wheels:
```text

pip install cmake
pip install dlib-19.24.1-cp39-cp39-win_amd64.whl
```

## 🧑‍💻 Usage
1️⃣ Encode Known Faces From Folder

Prepare a folder:
```text
known_faces/
 ├── john/
 │    ├── 1.jpg
 │    └── 2.jpg
 ├── mary/
      ├── img1.png
      └── pic2.jpg

```
# Run encoding script:
```text
from face_encoder_and_enroller import create_face_encodings

create_face_encodings(
    images_dir="known_faces",
    output_file="data/face_encodings.pkl"
)
```

## Outputs:
✔ Face encodings
✔ Name associations
✔ Saved .pkl dataset

##2️⃣ Enroll a New Face via Webcam

Run:
```bash

python "face encoder and enroller.py"

```
Or use programmatically:

```text

from face_encoder_and_enroller import enroll_face_from_camera

enroll_face_from_camera(
    name="John Doe",
    encodings_file="data/face_encodings.pkl"
)

```

Controls:
```text

▶ SPACE = capture face

❌ ESC = exit
```
## 3️⃣ Run Real-Time Recognition

Start recognition:

```python

python face_recognition_module.py --camera 0

```
## Use custom encodings:

```text
python face_recognition_module.py --encodings data/face_encodings.pkl
```
## 🎥 Example: Real-Time Recognition (API Use)
```bash
import cv2
from face_recognition_module import FaceRecognition

face_recog = FaceRecognition(
    encodings_file="data/face_encodings.pkl",
    confidence_threshold=0.6,
    processing_scale=0.5,
    process_every_n_frames=3
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = face_recog.recognize_faces(frame)
    cv2.imshow("Recognition", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_recog.cleanup()
```

### 🧩 Adding Face Encodings Programmatically
```python

import cv2
from face_recognition_module import FaceRecognition

face_recog = FaceRecognition()

image = cv2.imread("new_user.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_recog.add_face_encoding(rgb, "New User")
face_recog.save_encodings()
```

## 📊 Performance Features
✔ Frame Skipping

Processes frames every Nth frame for maximum FPS.

✔ Caching

Intermediate frames reuse last known detection → smoother tracking.

✔ Multithreaded

Thread pool improves real-time performance.

✔ Scaled Processing

Processing lower resolution for speed, while preserving accurate results.

✔ Statistics Overlay

Automatically displays:

FPS

Faces detected

Frames processed

Current frame number

## 🛡 Troubleshooting
dlib fails to install

Use prebuilt wheels for your OS and Python version.

No face detected

Make sure:

The face is well-lit

Avoid profile angles

Avoid masks blocking facial landmarks

Low FPS

Try:
```text

processing_scale=0.5 or 0.4

process_every_n_frames=3

Use HOG model (already default)

Black screen / no camera
```

Change camera index:
```python
python face_recognition_module.py --camera 1
```
## 🚀 Roadmap

Planned upgrades:

GPU acceleration (CUDA / cuDNN)

REST API server for remote recognition

Face metadata database (age, gender, emotions)

Multi-camera recognition pipeline

Web dashboard for management

## 🤝 Contribution

Fork the repo

Create a feature branch

Follow clean code practices

Submit a pull request

Pull requests are welcome!

## 📜 License

This project is released under the MIT License
You are free to use it in commercial and personal projects.

## 💬 Support

### If you encounter any issue or want new features added, feel free to open a GitHub issue.
