# 🎨 AI-Powered Virtual Air Canvas & Math Solver

A multimodal **"Neuro-Symbolic" AI application** that turns your webcam into a touchless digital whiteboard. It allows users to draw in the air using hand gestures, control the interface via voice commands, and **solve handwritten math equations instantly** using Deep Learning.

---

## 📸 Project Demo & Output

This system effectively handles arithmetic operations including Addition, Subtraction, Multiplication, Division, and Modulus.

| **Multiplication** | **Subtraction** | **Addition** |
|:---:|:---:|:---:|
| ![Multiplication](virtual_air_canvas/Output1.png) | ![Subtraction](virtual_air_canvas/Output4.png) | ![Addition](virtual_air_canvas/Output5.png) |
| *5 x 4 = 20* | *5 - 7 = -2* | *7 + 1 = 8* |

| **Division** | **Modulus** |
|:---:|:---:|
| ![Division](virtual_air_canvas/Output3.png) | ![Modulus](virtual_air_canvas/Output2.png) |
| *7 / 3 = 2.33* | *2 % 2 = 0* |

---

## 🌟 Significance & Real-World Impact

This project is not just a drawing tool; it is a step towards **Touchless Human-Computer Interaction (HCI)**. By combining Computer Vision with Speech Recognition, we eliminate the need for physical peripherals like mice or keyboards.

### Applications:

1. **Interactive Education:** Teachers can solve problems on a projected screen using only gestures, making learning more engaging.
2. **Sterile Environments:** In operating rooms, surgeons can control digital interfaces without touching unsterile screens.
3. **Accessibility:** Provides a computer interface for individuals with limited motor control who cannot use a physical mouse/keyboard.
4. **Smart Mirrors/Glass:** Can be integrated into AR displays where physical input methods are impossible.

---

## 🛠️ Methodology & Tech Stack

This project uses a **Neuro-Symbolic** approach:

* **"Neuro" (Neural Networks):** Uses Deep Learning (EasyOCR/CRNN) to *perceive* and read the handwritten text.
* **"Symbolic" (Logic):** Uses Python's symbolic logic to *calculate* the exact result, avoiding the "hallucinations" common in Large Language Models.

### Core Components:

1. **Hand Tracking (MediaPipe):**
   * Detects 21 hand landmarks in real-time.
   * **Logic:** If *Index Finger* is up → **Draw**. If *Index + Middle Finger* are up → **Hover/Select**.

2. **Jitter Reduction (One Euro Filter):**
   * Raw hand movement is shaky. We implemented the **One Euro Filter** (used in VR gaming) to smooth the signal mathematically, creating professional-grade strokes.

3. **Voice Control (Vosk):**
   * An offline, privacy-focused speech recognition engine.
   * We restricted the grammar to specific keywords (`"Solve"`, `"Clear"`, `"Red"`, etc.) to achieve **<200ms latency**.

4. **Math Solving (EasyOCR + RegEx):**
   * Captures the canvas, uses a Convolutional Recurrent Neural Network (CRNN) to read the digits, cleans the output using Regex, and computes the result via Python.

---

## 📂 Project Structure

```text
Virtual_Air_Canvas/
│
├── .venv/                         # Virtual Environment (Dependencies)
│
├── models/                        # Offline AI Models
│   ├── vosk-model-small-en-us.../ # Speech Recognition Model
│   └── hand_landmarker.task       # MediaPipe Tracking Model
│
├── src/                           # Source Code
│   ├── __init__.py
│   └── main.py                    # Core Logic (Vision + Voice + AI)
│
├── screenshots/                   # Output images for README
├── .gitignore                     # Git configuration
├── requirements.txt               # Dependency list
└── README.md                      # Documentation
```

---

## 🚀 How to Run

### Prerequisites

* Python 3.10 or 3.11 (Recommended)
* A Webcam
* A Microphone
* NVIDIA GPU (Optional, but recommended for faster OCR)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/PranavVaish/Virtual-Air-Canvas.git
   cd Virtual-Air-Canvas
   ```

2. **Create Virtual Environment:**

   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: If you have an NVIDIA GPU, ensure you install the CUDA-enabled version of PyTorch.*

4. **Download Models:**

   * Download the **Vosk Small English Model** and place it in the `models/` folder.
   * The **EasyOCR** model will download automatically on the first run.

5. **Run the App:**

   ```bash
   python src/main.py
   ```

---

## 🎮 User Guide

### 1. Gestures

* **Draw:** Lift **only** your Index Finger.
* **Move Cursor (Hover):** Lift **both** Index and Middle Fingers.
* **Stop/Idle:** Close your fist or drop your hand.

### 2. Voice Commands

Speak clearly into the microphone. The system is optimized for these specific words:

| Command | Action |
|:--------|:-------|
| **"Red"** | Change ink color to Red. |
| **"Blue"** | Change ink color to Blue. |
| **"Green"** | Change ink color to Green. |
| **"Black"** | Activates **Eraser Mode** (Huge brush size). |
| **"Clear"** | Wipes the entire screen clean. |
| **"Solve"** | AI reads the screen and prints the answer. |

---

## 🔧 Troubleshooting

* **Issue: "Vosk model not found"**
  * *Fix:* Ensure the folder structure in `models/` matches the path defined in `main.py`.

* **Issue: Camera lag**
  * *Fix:* Ensure you are running in a well-lit room. The code is optimized for GPU, but lighting affects the webcam exposure rate.

* **Issue: Math not reading correctly**
  * *Fix:* Draw large, clear numbers. Ensure there is good contrast between the background (your wall) and the drawing color.

---

## 👥 Credits & Developers

This project was architected and developed by:

**Pranav Vaish** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pranavvaish20)  
*Specialization: Computer Vision, AI Pipeline Integration, and System Architecture.*

**Dikshant Arora** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dikshant-arora-794149323)  
*Specialization: Speech Recognition, Optimization Algorithms, and Logic Implementation.*

---

## 📄 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## ⭐ Show your support

Give a ⭐️ if this project helped you!