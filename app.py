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

HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(gray):
    faces = HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    return faces if len(faces)>0 else []

class RoundedFrame(tk.Canvas):
    def __init__(self, parent, radius=16, bg=SURFACE, border=BORDER, **kw):
        super().__init__(parent, bg=BG, highlightthickness=0, **kw)
        self._radius = radius
        self._bg     = bg
        self._border = border
        self.bind("<Configure>", self._redraw)
    
    def _redraw(self, evt=None):
        w, h = self.winfo_width(), self.winfo_height()
        r    = self._radius
        self.delete("bg")
        pts = [r,0, w-r,0, w,0, w,r, w,h-r, w,h, w-r,h, r,h, 0,h, 0,h-r, 0,r, 0,0]
        self.create_polygon(pts, smooth=True, fill=self._bg, outline=self._border, width=1, tags="bg")
    
class Sidebar(tk.Frame):
    def __init__(self, parent, on_nav, **kw):
        super().__init__(parent, bg=SURFACE, width=200, **kw)
        self.pack_propagate(False)
        self._on_nav = on_nav
        self._buttons = {}
        self._active  = None
        self._build()    
    
    def _build(self):
        # Logo
        logo = tk.Frame(self, bg=SURFACE)
        logo.pack(fill=tk.X, padx=16, pady=(20,12))
        tk.Label(logo, text="🧠", font=("Segoe UI",28), bg=SURFACE, fg=TEXT).pack()
        tk.Label(logo, text="EmotiScan AI", font=("Segoe UI",13,"bold"), bg=SURFACE, fg=TEXT).pack()
        tk.Label(logo, text="v3.0 · Tkinter Edition", font=FONT_SMALL, bg=SURFACE, fg=MUTED).pack()

        tk.Frame(self, height=1, bg=BORDER).pack(fill=tk.X, padx=12, pady=8)

        nav_items = [
            ("📷 Image Upload",   "image"),
            ("🎥 Live Webcam",    "webcam"),
            ("🏋 Train Model",   "train"),
            ("☁ Google Colab",   "colab"),
            ("📊 Model Info",     "info"),
        ]
        for label, key in nav_items:
            btn = tk.Button(
                self, text=label, anchor="w",
                font=FONT_BODY, bd=0, relief=tk.FLAT, cursor="hand2",
                padx=16, pady=10,
                bg=SURFACE, fg=TEXT, activebackground=ACCENT,
                activeforeground=WHITE, command=lambda k=key: self._nav(k)
            )
            btn.pack(fill=tk.X, padx=8, pady=2)
            self._buttons[key] = btn

        tk.Frame(self, height=1, bg=BORDER).pack(fill=tk.X, padx=12, pady=8)

        # Settings label
        tk.Label(self, text="SETTINGS", font=("Segoe UI",8,"bold"),
                 bg=SURFACE, fg=MUTED).pack(anchor="w", padx=16)

        # Confidence slider
        tk.Label(self, text="Min Confidence", font=FONT_SMALL, bg=SURFACE, fg=TEXT).pack(anchor="w", padx=16, pady=(8,0))
        self.conf_var = tk.IntVar(value=30)
        sl = ttk.Scale(self, from_=0, to=100, variable=self.conf_var, orient=tk.HORIZONTAL)
        sl.pack(fill=tk.X, padx=16)
        tk.Label(self, textvariable=self.conf_var, font=FONT_SMALL, bg=SURFACE, fg=ACCENT).pack(anchor="w", padx=16)

        # Enhance toggle
        self.enhance_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Pre-enhance image", variable=self.enhance_var,
                       font=FONT_SMALL, bg=SURFACE, fg=TEXT,
                       selectcolor=SURFACE2, activebackground=SURFACE).pack(anchor="w", padx=16, pady=4)

        # Show bars toggle
        self.bars_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Show probability bars", variable=self.bars_var,
                       font=FONT_SMALL, bg=SURFACE, fg=TEXT,
                       selectcolor=SURFACE2, activebackground=SURFACE).pack(anchor="w", padx=16, pady=4)

        # Bottom status
        self.status_lbl = tk.Label(self, text="⚫ No model loaded",
                                   font=FONT_SMALL, bg=SURFACE, fg=RED, wraplength=180)
        self.status_lbl.pack(side=tk.BOTTOM, padx=8, pady=12)
    
    def _nav(self, key):
        for k, b in self._buttons.items():
            b.config(bg=SURFACE if k != key else ACCENT, fg=TEXT if k != key else WHITE)
        self._active = key
        self._on_nav(key)
    
    def set_model_status(self, loaded):
        if loaded:
            self.status_lbl.config(text="🟢 Model loaded", fg=GREEN)
        else:
            self.status_lbl.config(text="🔴 No model loaded", fg=RED)

    def activate(self, key):
        self._nav(key)

if __name__ == "__main__":
    app = EmotiScanApp()
    app.mainloop()