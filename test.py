# --- 1. GPU MODEL LOADER (The "Brain") ---
import torch
print(f"🚀 Loading AI Model to RTX 3050 (4GB)...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Hardware Detected: {DEVICE}")
