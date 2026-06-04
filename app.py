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

class PageBase(tk.Frame):
    def __init__(self, parent, app, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.app = app
    
    def heading(self, text):
        tk.Label(self, text=text, font=FONT_TITLE, bg=BG, fg=TEXT).pack(
            anchor="w", padx=24, pady=(20,4))
        tk.Frame(self, height=1, bg=BORDER).pack(fill=tk.X, padx=24)
    
    def card(self, parent, title=None, accent=False):
        outer = tk.Frame(parent, bg=BORDER, bd=0)
        inner = tk.Frame(outer, bg=SURFACE, padx=16, pady=14)
        inner.pack(padx=1, pady=1, fill=tk.BOTH, expand=True)
        if accent:
            bar = tk.Frame(inner, bg=ACCENT, width=4)
            bar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,12))
        if title:
            tk.Label(inner, text=title, font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w")
        return inner, outer

    def accent_btn(self, parent, text, command, color=ACCENT):
        return tk.Button(parent, text=text, command=command,
                         font=("Segoe UI",10,"bold"), bd=0, relief=tk.FLAT, cursor="hand2",
                         bg=color, fg=WHITE, activebackground=color, padx=18, pady=8)

class ImagePage(PageBase):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.heading("📷  Image Upload & Detection")
        self._result_img = None
        self._build()

    def _build(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=12)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left: input
        left_card, left_outer = self.card(body, accent=True)
        left_outer.grid(row=0, column=0, sticky="nsew", padx=(0,8), pady=4)
        tk.Label(left_card, text="Input Image", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w", pady=(0,8))

        self._inp_lbl = tk.Label(left_card, bg=SURFACE2, text="Drop image here or click Upload",
                                 fg=MUTED, font=FONT_BODY, width=40, height=18, relief="flat",
                                 cursor="hand2")
        self._inp_lbl.pack(fill=tk.BOTH, expand=True)
        self._inp_lbl.bind("<Button-1>", lambda e: self._upload())

        btn_row = tk.Frame(left_card, bg=SURFACE)
        btn_row.pack(fill=tk.X, pady=(10,0))
        self.accent_btn(btn_row, "📂  Upload Image", self._upload).pack(side=tk.LEFT)
        self.accent_btn(btn_row, "🔍  Detect", self._detect, color="#2563eb").pack(side=tk.LEFT, padx=(8,0))

        self._inp_info = tk.Label(left_card, text="", font=FONT_SMALL, bg=SURFACE, fg=MUTED)
        self._inp_info.pack(anchor="w", pady=(4,0))

        # Right: output
        right_card, right_outer = self.card(body, accent=True)
        right_outer.grid(row=0, column=1, sticky="nsew", padx=(8,0), pady=4)
        tk.Label(right_card, text="Detection Result", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w", pady=(0,8))

        self._out_lbl = tk.Label(right_card, bg=SURFACE2, text="Results will appear here",
                                 fg=MUTED, font=FONT_BODY, width=40, height=18)
        self._out_lbl.pack(fill=tk.BOTH, expand=True)

        self._out_info = tk.Label(right_card, text="", font=FONT_SMALL, bg=SURFACE, fg=MUTED)
        self._out_info.pack(anchor="w", pady=(4,0))

        # Probability bars panel
        bar_card, bar_outer = self.card(body)
        bar_outer.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8,0))
        tk.Label(bar_card, text="Emotion Probabilities", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w")
        self._bars_frame = tk.Frame(bar_card, bg=SURFACE)
        self._bars_frame.pack(fill=tk.X, pady=(8,0))
        tk.Label(self._bars_frame, text="Run detection to see probabilities",
                 font=FONT_SMALL, fg=MUTED, bg=SURFACE).pack()

        self._pil_img = None
    
    def _upload(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images","*.jpg *.jpeg *.png *.webp *.bmp")])
        if not path:
            return
        self._img_path = path
        img = Image.open(path).convert("RGB")
        self._pil_img = img
        tk_img = self._fit_image(img, (self._inp_lbl.winfo_width() or 400,
                                       self._inp_lbl.winfo_height() or 320))
        self._inp_lbl.config(image=tk_img, text="")
        self._inp_lbl._img = tk_img  # prevent GC
        self._inp_info.config(text=f"{img.width}×{img.height}px  ·  {Path(path).name}")
        self._out_lbl.config(image="", text="Click Detect →")
    
    def _detect(self):
        if not self._pil_img:
            messagebox.showwarning("No image", "Please upload an image first.")
            return
        if not self.app.model:
            messagebox.showwarning("No model", "No model loaded.\nPlease train a model first (Train Model tab).")
            return

        np_img = np.array(self._pil_img)
        enhance = self.app.sidebar.enhance_var.get()
        if enhance:
            gray = cv2.equalizeHist(cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY))
        else:
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        faces = detect_faces(gray)
        result_img = self._pil_img.copy()
        draw = ImageDraw.Draw(result_img)
        all_results = []

        if len(faces) == 0:
            self._out_info.config(text="⚠️ No face detected. Try a clearer image.", fg=YELLOW)
        else:
            min_conf = self.app.sidebar.conf_var.get()
            for (x,y,w,h) in faces:
                tensor = preprocess_face(gray,x,y,w,h)
                emotion, conf, all_p = predict_emotion(self.app.model, tensor, self.app.labels)
                if conf >= min_conf:
                    color = EMOTION_COLORS.get(emotion, "#7c6af7")
                    draw.rectangle([x,y,x+w,y+h], outline=color, width=3)
                    lbl = f"{EMOTION_EMOJI.get(emotion,'')} {emotion} {conf:.0f}%"
                    draw.rectangle([x, max(0,y-26), x+len(lbl)*9, y], fill=color)
                    draw.text((x+4, max(0,y-22)), lbl, fill="white")
                    all_results.append((emotion, conf, all_p))

            self._out_info.config(
                text=f"✅ {len(all_results)} face(s) detected", fg=GREEN)
            self._show_bars(all_results[0][2] if all_results else {})

        tk_img = self._fit_image(result_img, (self._out_lbl.winfo_width() or 400,
                                               self._out_lbl.winfo_height() or 320))
        self._out_lbl.config(image=tk_img, text="")
        self._out_lbl._img = tk_img
    
    def _show_bars(self, all_probs):
        for w in self._bars_frame.winfo_children():
            w.destroy()
        if not all_probs:
            return
        sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
        for emotion, pct in sorted_probs:
            color = EMOTION_COLORS.get(emotion, ACCENT)
            row = tk.Frame(self._bars_frame, bg=SURFACE)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"{EMOTION_EMOJI.get(emotion,'')} {emotion:<10}",
                     font=FONT_SMALL, bg=SURFACE, fg=TEXT, width=14, anchor="w").pack(side=tk.LEFT)
            bar_bg = tk.Frame(row, bg=SURFACE2, height=14)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True)
            bar_bg.update_idletasks()
            bar_w = int((pct/100) * (bar_bg.winfo_width() or 300))
            tk.Frame(bar_bg, bg=color, width=bar_w, height=14).place(x=0,y=0)
            tk.Label(row, text=f"{pct:5.1f}%", font=("Consolas",9),
                     bg=SURFACE, fg=TEXT, width=6).pack(side=tk.LEFT, padx=(4,0))
    
    def _fit_image(self, img, box):
        bw, bh = max(box[0],1), max(box[1],1)
        img.thumbnail((bw, bh), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

class WebcamPage(PageBase):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._running = False
        self._thread  = None
        self._cap     = None
        self.heading("🎥  Live Webcam Detection")
        self._build()

if __name__ == "__main__":
    app = EmotiScanApp()
    app.mainloop()