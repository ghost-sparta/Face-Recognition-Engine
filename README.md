# Face-Recognition-Engine
📌 Face Recognition Engine — Encoder, Enroller & Real-Time Recognition

A complete, optimized, and production-ready face recognition framework built using Python, OpenCV, and face_recognition.
This project includes:

Face Encoder & Enrollment Tool — Register new users from images or webcam

Optimized Recognition Engine — Multi-threaded, scalable, and designed for real-world deployment

High-performance real-time detection

Full statistics monitoring and caching system

🚀 Features
✔ Face Encoder & Enrollment

Encode faces from folder structure: known_faces/person_name/*.jpg

Add new face via webcam with realtime preview

Automatically saves encodings to .pkl files

✔ Production Face Recognition Engine

Thread-pooled processing

Frame skipping (process every Nth frame)

Optimized RGB conversion & resolution scaling

Confidence scoring

Smart caching to increase FPS

Automatic statistics (FPS, faces, frames processed)

Robust encoding loader with backward compatibility

Safe read/write operations for encodings

✔ Flexible & Extensible

Plug into CCTV systems

Integrate with APIs

Supports custom pipelines (PyQt, Flask, FastAPI, etc.)
