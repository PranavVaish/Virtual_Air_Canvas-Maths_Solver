import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import queue
import pyaudio
import json
import os
import re
import easyocr  # <--- The New CRNN Algorithm
from vosk import Model, KaldiRecognizer
from PIL import Image

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOSK_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "../models/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"))
MP_TASK_PATH = os.path.normpath(os.path.join(BASE_DIR, "../models/hand_landmarker.task"))

# --- 1. LOAD CRNN MODEL (EasyOCR) ---
print(f"🚀 Loading CRNN (EasyOCR) to GPU...")
# This downloads a small efficient model trained for reading text
reader = easyocr.Reader(['en'], gpu=True) 
print("✅ Math Reader Ready!")

# --- 2. SMOOTHING FILTER ---
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

# --- 3. VOICE THREAD ---
command_queue = queue.Queue()

# --- REPLACEMENT VOICE FUNCTION ---
def voice_listener():
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"❌ Error: Vosk model not found at {VOSK_MODEL_PATH}")
        return
    
    from vosk import SetLogLevel
    SetLogLevel(-1)
    
    try:
        model = Model(VOSK_MODEL_PATH)
        
        # --- FIX: GRAMMAR RESTRICTION ---
        # We tell Vosk to ONLY listen for these specific words. 
        # This makes recognition 10x faster and fixes the "Solve" issue.
        # "[unk]" stands for "Unknown" (noise).
        grammar = '["red", "blue", "green", "black", "clear", "solve", "[unk]"]'
        rec = KaldiRecognizer(model, 16000, grammar)
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        
        print("🎤 Voice Engine: Optimized (Listening only for commands)")
        
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            
            # Using PartialResult can be faster, but with Grammar, Result is safer
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result['text']
                
                # Simple keyword matching
                if "solve" in text: command_queue.put("SOLVE") # Check this first!
                elif "red" in text: command_queue.put("RED")
                elif "blue" in text: command_queue.put("BLUE")
                elif "green" in text: command_queue.put("GREEN")
                elif "black" in text: command_queue.put("BLACK")
                elif "clear" in text: command_queue.put("CLEAR")
                
    except Exception as e:
        print(f"🎤 Voice Error: {e}")

# --- 4. MATH HELPER ---
def safe_calculate(expression):
    try:
        # Common OCR mistakes fix
        expression = expression.lower().replace('x', '*') # Convert 'x' to multiply
        expression = expression.replace('s', '5')         # S -> 5
        expression = expression.replace('o', '0')         # O -> 0
        expression = expression.replace('z', '2')         # Z -> 2
        
        # --- NEW: Support for Powers ---
        # Python uses '**' for power, but humans write '^'. 
        # We swap them here.
        expression = expression.replace('^', '**')
        # Clean: Keep only numbers and operators
        cleaned_expr = re.sub(r'[^0-9\.\+\-\*\/\%\(\)]', '', expression)
        print(f"🧮 Parsed Equation: {cleaned_expr}")
        
        if not cleaned_expr: return "No Math Found"

        # Calculate
        result = eval(cleaned_expr)
        
        # Format
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(round(result, 2))
        
    except ZeroDivisionError:
        return "Div by 0"
    except Exception as e:
        return expression

# --- 5. MAIN LOOP ---
def main():
    t = threading.Thread(target=voice_listener, daemon=True)
    t.start()

    mp_hands = mp.solutions.hands # type: ignore
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, model_complexity=0)
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) 
    cap.set(4, 480)
    
    canvas = np.zeros((480, 640, 3), np.uint8)
    
    x_filter = OneEuroFilter(time.time(), 0, min_cutoff=0.01, beta=0.1)
    y_filter = OneEuroFilter(time.time(), 0, min_cutoff=0.01, beta=0.1)
    xp, yp = 0, 0
    
    draw_color = (0, 0, 255) # Start with Red
    brush_thickness = 10
    ai_read_text = ""
    last_answer = ""
    
    print("📸 System Ready. Draw & Speak!")

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        # --- COMMAND HANDLER ---
        try:
            cmd = command_queue.get_nowait()
            print(f"🎤 Command: {cmd}")
            if cmd == "RED": 
                draw_color = (0, 0, 255)
                brush_thickness = 10
            elif cmd == "BLUE": 
                draw_color = (255, 0, 0)
                brush_thickness = 10
            elif cmd == "GREEN": 
                draw_color = (0, 255, 0)
                brush_thickness = 10
            elif cmd == "BLACK": 
                draw_color = (0, 0, 0)    # Actual Black Color
                brush_thickness = 50      # Huge Eraser
                print("🧽 Eraser Mode Active")
            elif cmd == "CLEAR": 
                canvas = np.zeros((h, w, 3), np.uint8)
                last_answer = ""
                ai_read_text = ""
            elif cmd == "SOLVE":
                print("🧠 CRNN Reading...")
                # Save temp image for OCR
                cv2.imwrite("math_temp.png", canvas)
                
                try:
                    # Run EasyOCR (CRNN) on the saved image
                    results = reader.readtext("math_temp.png", detail=0)
                    
                    # Normalize OCR output to a flat string (handle various return shapes)
                    text_parts = []
                    for r in results:
                        if isinstance(r, str):
                            text_parts.append(r)
                        elif isinstance(r, (list, tuple)) and r:
                            text_parts.append(str(r[0]))
                        elif isinstance(r, dict) and 'text' in r:
                            text_parts.append(str(r['text']))
                        else:
                            text_parts.append(str(r))
                    ai_read_text = "".join(text_parts)
                    print(f"👁️ OCR Read: {ai_read_text}")
                    
                    last_answer = safe_calculate(ai_read_text)
                    print(f"🤖 Result: {last_answer}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except queue.Empty: pass

        # --- HAND TRACKING ---
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            raw_x = int(lm.landmark[8].x * w)
            raw_y = int(lm.landmark[8].y * h)
            x1 = int(x_filter(time.time(), raw_x))
            y1 = int(y_filter(time.time(), raw_y))
            
            index_up = lm.landmark[8].y < lm.landmark[6].y
            middle_up = lm.landmark[12].y < lm.landmark[10].y
            
            # Draw Mode
            if index_up and not middle_up:
                if xp == 0 and yp == 0: xp, yp = x1, y1
                
                # DRAW LINE
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                xp, yp = x1, y1
            # Hover Mode
            elif index_up and middle_up:
                xp, yp = 0, 0
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            else:
                xp, yp = 0, 0

        # --- MASKING LOGIC (THE BLACK COLOR FIX) ---
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        
        # Any pixel that is NOT black ( > 10 brightness) is treated as drawing
        _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
        
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        
        # 1. Black out the area in webcam where drawing exists
        img = cv2.bitwise_and(img, imgInv)
        
        # 2. Add the colored drawing on top
        # Note: If draw_color was (0,0,0), it added nothing to 'canvas' brightness
        # So the threshold kept it as 'background', effectively erasing it!
        img = cv2.bitwise_or(img, canvas)

        # UI Text
        if last_answer:
             cv2.putText(img, f"Read: {ai_read_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
             cv2.putText(img, f"= {last_answer}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("RTX Air Canvas", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()