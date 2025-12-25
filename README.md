# Face Privacy Filter (Python + OpenCV)

A real-time face privacy filter that detects faces from a webcam feed and applies **blur or pixelation** to protect identity. Suitable for demos, privacy-aware applications, and computer vision learning projects.

---

 Features

- Real-time face detection using Haar Cascade
- Blur and Pixelation modes
- Adjustable filter strength
- Keyboard controls
- Simple on-screen UI
- Lightweight and fast
- Cross-platform support

---

Technologies Used

- Python 3.x
- OpenCV
- NumPy

---

 Installation

1. Clone the repository

bash
git clone https://github.com/shanawazcoder/face-privacy-filter-opencv.git
cd face-privacy-filter-opencv

2. Install dependencies
pip install opencv-python numpy

Usage

Run with default settings:

python privacy_filter.py


Run with custom mode and strength:

python privacy_filter.py --mode blur --strength 25

Keyboard Controls
Key	Action
P	Toggle Blur / Pixelate
+	Increase filter strength
-	Decrease filter strength
Q	Quit application
 How It Works

Captures video frames from webcam

Converts frames to grayscale

Detects faces using Haar Cascade

Applies blur or pixelation to detected faces

Displays processed video in real time

Project Structure
face-privacy-filter-opencv/
│
├── main.py
├── README.md
├── requirements.txt


Use Cases

Webcam privacy protection

Public video anonymization

Computer vision practice

CCTV or monitoring prototypes

YouTube or streaming face hiding
