import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import json
import time
import io
import webbrowser
import subprocess
import queue
import urllib.request
import urllib.parse
import zipfile
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2


BG        = "#0d0f14"
SURFACE   = "#151820"
SURFACE2  = "#1c2030"
BORDER    = "#252a35"
ACCENT    = "#7c6af7"
ACCENT2   = "#f76aaa"
GREEN     = "#4ade80"
RED       = "#ef4444"
YELLOW    = "#f59e0b"
TEXT      = "#e8ecf5"
MUTED     = "#6b7280"
WHITE     = "#ffffff"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEAD   = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_EMOJI = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😄", "Sad": "😢", "Surprise": "😮", "Neutral": "😐",
}
EMOTION_COLORS = {
    "Angry": "#ef4444", "Disgust": "#84cc16", "Fear": "#8b5cf6",
    "Happy": "#f59e0b", "Sad": "#3b82f6", "Surprise": "#f97316", "Neutral": "#6b7280",
}
# BGR for OpenCV
EMOTION_BGR = {
    "Angry": (60,60,239), "Disgust": (55,200,100), "Fear": (180,90,220),
    "Happy": (30,165,245), "Sad": (235,100,55), "Surprise": (30,155,250), "Neutral": (130,120,100),
}

BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "model"
DATA_DIR   = BASE_DIR / "dataset"
MODEL_PATH = MODEL_DIR / "emotion_model.h5"
META_PATH  = MODEL_DIR / "model_meta.json"


def hex_to_bgr(h):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b,g,r)

def load_model_and_labels():
    DEFAULT = EMOTIONS[:]
    if not MODEL_PATH.exists():
        return None, DEFAULT
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH))
        if META_PATH.exists():
            with open(META_PATH) as f:
                meta = json.load(f)
            labels = meta.get("labels", DEFAULT)
        else:
            labels = DEFAULT
        return model, labels
    except Exception as e:
        print(f"Model load error: {e}")
        return None, DEFAULT
    
def preprocess_face(gray, x, y, w, h):
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48,48)).astype("float32") / 255.0
    return roi.reshape(1,48,48,1)    

def predict_emotion(model, tensor, labels):
    probs = model.predict(tensor, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = labels[idx]
    conf  = float(probs[idx]) * 100
    all_p = {labels[i]: float(probs[i])*100 for i in range(len(labels))}
    return label, conf, all_p

if __name__ == "__main__":
    app = EmotiScanApp()
    app.mainloop()