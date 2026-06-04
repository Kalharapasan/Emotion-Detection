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
    
    def _build(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=12)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Video feed
        vid_card, vid_outer = self.card(body, accent=True)
        vid_outer.grid(row=0, column=0, sticky="nsew", padx=(0,8), pady=4)
        tk.Label(vid_card, text="Live Feed", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w")
        self._video_lbl = tk.Label(vid_card, bg=SURFACE2, text="Camera feed will appear here",
                                   fg=MUTED, font=FONT_BODY)
        self._video_lbl.pack(fill=tk.BOTH, expand=True, pady=(8,0))

        # Controls & stats
        ctrl_card, ctrl_outer = self.card(body)
        ctrl_outer.grid(row=0, column=1, sticky="nsew", padx=(8,0), pady=4)
        tk.Label(ctrl_card, text="Controls", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w", pady=(0,12))

        self.start_btn = self.accent_btn(ctrl_card, "▶  Start Camera", self._start, GREEN)
        self.start_btn.pack(fill=tk.X, pady=3)
        self.stop_btn = self.accent_btn(ctrl_card, "⏹  Stop Camera", self._stop, RED)
        self.stop_btn.pack(fill=tk.X, pady=3)
        self.accent_btn(ctrl_card, "📷  Save Frame", self._save_frame, "#2563eb").pack(fill=tk.X, pady=3)

        tk.Frame(ctrl_card, height=1, bg=BORDER).pack(fill=tk.X, pady=10)
        tk.Label(ctrl_card, text="Live Stats", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w")

        self._stat_frame = tk.Label(ctrl_card, text="#", font=FONT_MONO, bg=SURFACE, fg=MUTED)
        self._stat_frame.pack(anchor="w", pady=2)
        self._stat_faces = tk.Label(ctrl_card, text="Faces: 0", font=FONT_MONO, bg=SURFACE, fg=TEXT)
        self._stat_faces.pack(anchor="w")
        self._stat_fps   = tk.Label(ctrl_card, text="FPS: 0", font=FONT_MONO, bg=SURFACE, fg=TEXT)
        self._stat_fps.pack(anchor="w")
        self._stat_emo   = tk.Label(ctrl_card, text="Dominant: —", font=FONT_MONO, bg=SURFACE, fg=ACCENT, wraplength=160)
        self._stat_emo.pack(anchor="w", pady=2)

        tk.Frame(ctrl_card, height=1, bg=BORDER).pack(fill=tk.X, pady=10)
        tk.Label(ctrl_card, text="Session Counts", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w")
        self._count_labels = {}
        for emo in EMOTIONS:
            row = tk.Frame(ctrl_card, bg=SURFACE)
            row.pack(fill=tk.X, pady=1)
            color = EMOTION_COLORS.get(emo, ACCENT)
            tk.Label(row, text=f"{EMOTION_EMOJI.get(emo,'')} {emo:<9}",
                     font=FONT_SMALL, bg=SURFACE, fg=TEXT, width=14, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="0", font=("Consolas",9), bg=SURFACE, fg=color)
            lbl.pack(side=tk.RIGHT)
            self._count_labels[emo] = lbl

        self._frame_to_show = None
        self._emo_counts   = {e: 0 for e in EMOTIONS}
        self._saved_frame  = None
    
    def _stop(self):
        self._running = False

    def _save_frame(self):
        if self._saved_frame is not None:
            path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                filetypes=[("JPEG","*.jpg")])
            if path:
                cv2.imwrite(path, self._saved_frame)
                messagebox.showinfo("Saved", f"Frame saved to:\n{path}")
        
    
    def _loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open camera.")
            self._running = False
            return
        fc = 0
        t0 = time.time()
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            fc += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)
            min_conf = self.app.sidebar.conf_var.get()
            for (x,y,w,h) in faces:
                tensor = preprocess_face(gray,x,y,w,h)
                emotion, conf, _ = predict_emotion(self.app.model, tensor, self.app.labels)
                self._emo_counts[emotion] = self._emo_counts.get(emotion,0)+1
                if conf >= min_conf:
                    col = EMOTION_BGR.get(emotion,(150,150,150))
                    cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
                    txt = f"{emotion} {conf:.0f}%"
                    (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(frame,(x,max(0,y-th-10)),(x+tw+8,y),col,-1)
                    cv2.putText(frame,txt,(x+4,max(th,y-4)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
            fps = fc / (time.time()-t0+1e-9)
            cv2.putText(frame,f"FPS:{fps:.1f}",(frame.shape[1]-90,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,255,100),1)
            self._saved_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frame_to_show = (rgb, fc, len(faces), fps)
        cap.release()
        self._running = False
    
    def _update_ui(self):
        if self._frame_to_show is not None:
            rgb, fc, nf, fps = self._frame_to_show
            h, w = rgb.shape[:2]
            lbl_w = max(self._video_lbl.winfo_width(), 400)
            lbl_h = max(self._video_lbl.winfo_height(), 300)
            scale = min(lbl_w/w, lbl_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            if new_w>0 and new_h>0:
                img = Image.fromarray(rgb).resize((new_w,new_h), Image.NEAREST)
                tk_img = ImageTk.PhotoImage(img)
                self._video_lbl.config(image=tk_img, text="")
                self._video_lbl._img = tk_img
            self._stat_frame.config(text=f"Frame: #{fc}")
            self._stat_faces.config(text=f"Faces: {nf}")
            self._stat_fps.config(text=f"FPS: {fps:.1f}")
            dom = max(self._emo_counts, key=self._emo_counts.get) if any(self._emo_counts.values()) else "—"
            self._stat_emo.config(text=f"Dominant: {EMOTION_EMOJI.get(dom,'')} {dom}")
            for emo, lbl in self._count_labels.items():
                lbl.config(text=str(self._emo_counts.get(emo,0)))
            self._frame_to_show = None
        if self._running:
            self.after(30, self._update_ui)
        else:
            self._video_lbl.config(image="", text="Camera stopped")

class TrainPage(PageBase):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.heading("🏋  Model Training")
        self._train_thread = None
        self._log_queue    = queue.Queue()
        self._build()

    def _build(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=12)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        # Dataset section
        ds_card, ds_outer = self.card(body, "Dataset Setup", accent=True)
        ds_outer.grid(row=0, column=0, sticky="ew", padx=(0,8), pady=4)

        # Dataset path
        tk.Label(ds_card, text="Dataset Path (train/test folders)", font=FONT_SMALL, bg=SURFACE, fg=MUTED).pack(anchor="w", pady=(6,2))
        row1 = tk.Frame(ds_card, bg=SURFACE)
        row1.pack(fill=tk.X)
        self._ds_var = tk.StringVar(value=str(DATA_DIR))
        tk.Entry(row1, textvariable=self._ds_var, font=FONT_MONO,
                 bg=SURFACE2, fg=TEXT, insertbackground=TEXT, bd=0,
                 highlightthickness=1, highlightcolor=ACCENT, highlightbackground=BORDER).pack(
            side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self.accent_btn(row1, "Browse", self._browse_ds, "#374151").pack(side=tk.LEFT, padx=(4,0))

        tk.Frame(ds_card, height=1, bg=BORDER).pack(fill=tk.X, pady=8)
        tk.Label(ds_card, text="Auto-Download FER-2013 via Kaggle API", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w")
        tk.Label(ds_card, text="Paste your kaggle.json content below:", font=FONT_SMALL, bg=SURFACE, fg=MUTED).pack(anchor="w", pady=(4,2))
        self._kaggle_txt = tk.Text(ds_card, height=3, font=FONT_MONO,
                                   bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                   bd=0, highlightthickness=1,
                                   highlightcolor=ACCENT, highlightbackground=BORDER)
        self._kaggle_txt.pack(fill=tk.X, pady=(0,6))
        self._kaggle_txt.insert("1.0", '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}')

        row2 = tk.Frame(ds_card, bg=SURFACE)
        row2.pack(fill=tk.X, pady=4)
        self.accent_btn(row2, "⬇ Download Dataset", self._download_dataset).pack(side=tk.LEFT)
        self._ds_status = tk.Label(row2, text="", font=FONT_SMALL, bg=SURFACE, fg=MUTED)
        self._ds_status.pack(side=tk.LEFT, padx=(12,0))

        # Training config
        cfg_card, cfg_outer = self.card(body, "Training Config", accent=True)
        cfg_outer.grid(row=0, column=1, sticky="ew", padx=(8,0), pady=4)

        params = [
            ("Epochs",      "epochs",      "60"),
            ("Batch Size",  "batch",       "64"),
            ("Image Size",  "img_size",    "48"),
            ("Learning Rate","lr",         "0.001"),
        ]
        self._param_vars = {}
        for label, key, default in params:
            row = tk.Frame(cfg_card, bg=SURFACE)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, font=FONT_SMALL, bg=SURFACE, fg=TEXT, width=14, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            self._param_vars[key] = var
            tk.Entry(row, textvariable=var, font=FONT_MONO,
                     bg=SURFACE2, fg=TEXT, insertbackground=TEXT, bd=0,
                     highlightthickness=1, highlightcolor=ACCENT, highlightbackground=BORDER,
                     width=10).pack(side=tk.LEFT, ipady=4)

        tk.Frame(cfg_card, height=1, bg=BORDER).pack(fill=tk.X, pady=8)
        self._aug_var = tk.BooleanVar(value=True)
        tk.Checkbutton(cfg_card, text="Enable data augmentation", variable=self._aug_var,
                       font=FONT_SMALL, bg=SURFACE, fg=TEXT,
                       selectcolor=SURFACE2, activebackground=SURFACE).pack(anchor="w")
        self._earlystop_var = tk.BooleanVar(value=True)
        tk.Checkbutton(cfg_card, text="Early stopping (patience 10)", variable=self._earlystop_var,
                       font=FONT_SMALL, bg=SURFACE, fg=TEXT,
                       selectcolor=SURFACE2, activebackground=SURFACE).pack(anchor="w")

        # Train buttons
        btn_row = tk.Frame(cfg_card, bg=SURFACE)
        btn_row.pack(fill=tk.X, pady=(10,0))
        self._train_btn = self.accent_btn(btn_row, "🚀  Start Local Training", self._start_training, GREEN)
        self._train_btn.pack(side=tk.LEFT)
        self._stop_btn = self.accent_btn(btn_row, "⏹  Stop", self._stop_training, RED)
        self._stop_btn.pack(side=tk.LEFT, padx=(8,0))
        self._stop_btn.config(state=tk.DISABLED)

        # Progress + log
        prog_card, prog_outer = self.card(body)
        prog_outer.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8,0))
        prog_top = tk.Frame(prog_card, bg=SURFACE)
        prog_top.pack(fill=tk.X)
        tk.Label(prog_top, text="Training Progress", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(side=tk.LEFT)
        self._epoch_lbl = tk.Label(prog_top, text="", font=FONT_SMALL, bg=SURFACE, fg=MUTED)
        self._epoch_lbl.pack(side=tk.RIGHT)

        self._progress = ttk.Progressbar(prog_card, mode="determinate", maximum=100)
        self._progress.pack(fill=tk.X, pady=6)

        # Metrics row
        met_row = tk.Frame(prog_card, bg=SURFACE)
        met_row.pack(fill=tk.X, pady=(0,8))
        self._met_loss = self._metric_box(met_row, "Loss", "—")
        self._met_acc  = self._metric_box(met_row, "Accuracy", "—")
        self._met_vlos = self._metric_box(met_row, "Val Loss", "—")
        self._met_vacc = self._metric_box(met_row, "Val Accuracy", "—")

        # Log console
        tk.Label(prog_card, text="Console Output", font=FONT_HEAD, bg=SURFACE, fg=TEXT).pack(anchor="w")
        self._console = scrolledtext.ScrolledText(
            prog_card, height=12, font=FONT_MONO,
            bg="#0a0c10", fg=GREEN, insertbackground=GREEN,
            state=tk.DISABLED, bd=0)
        self._console.pack(fill=tk.BOTH, expand=True, pady=(4,0))

        self._training = False
    
    def _metric_box(self, parent, label, value):
        box = tk.Frame(parent, bg=SURFACE2, padx=14, pady=8)
        box.pack(side=tk.LEFT, padx=4)
        tk.Label(box, text=label, font=FONT_SMALL, bg=SURFACE2, fg=MUTED).pack()
        val_lbl = tk.Label(box, text=value, font=("Segoe UI",14,"bold"), bg=SURFACE2, fg=ACCENT)
        val_lbl.pack()
        return val_lbl

    def _log(self, msg):
        self._log_queue.put(msg)

    def _flush_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._console.config(state=tk.NORMAL)
                self._console.insert(tk.END, msg + "\n")
                self._console.see(tk.END)
                self._console.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(200, self._flush_log)

    def on_show(self):
        self._flush_log()

    def _browse_ds(self):
        path = filedialog.askdirectory()
        if path:
            self._ds_var.set(path)
        
    def _download_dataset(self):
        kaggle_json = self._kaggle_txt.get("1.0", tk.END).strip()
        try:
            cred = json.loads(kaggle_json)
        except:
            messagebox.showerror("Error", "Invalid kaggle.json format.\nExpected: {\"username\":\"...\",\"key\":\"...\"}")
            return

        dest = Path(self._ds_var.get())
        self._ds_status.config(text="Downloading...", fg=YELLOW)
        self.update_idletasks()

        def _do_download():
            try:
                kaggle_dir = Path.home() / ".kaggle"
                kaggle_dir.mkdir(exist_ok=True)
                with open(kaggle_dir / "kaggle.json", "w") as f:
                    json.dump(cred, f)
                os.chmod(kaggle_dir / "kaggle.json", 0o600)

                dest.mkdir(parents=True, exist_ok=True)
                self._log("📥 Starting FER-2013 download via Kaggle API...")
                result = subprocess.run(
                    [sys.executable, "-m", "kaggle", "datasets", "download",
                     "-d", "msambare/fer2013", "--unzip", "-p", str(dest)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    self._log("✅ Dataset downloaded successfully!")
                    self.after(0, lambda: self._ds_status.config(text="✅ Downloaded!", fg=GREEN))
                else:
                    self._log(f"❌ Kaggle error:\n{result.stderr}")
                    self.after(0, lambda: self._ds_status.config(text="❌ Failed", fg=RED))
            except Exception as e:
                self._log(f"❌ Error: {e}")
                self.after(0, lambda: self._ds_status.config(text="❌ Error", fg=RED))

        threading.Thread(target=_do_download, daemon=True).start()
    
    def _start_training(self):
        ds_path = Path(self._ds_var.get())
        train_dir = ds_path / "train"
        test_dir  = ds_path / "test"

        if not train_dir.exists():
            messagebox.showerror("Error", f"train/ folder not found in:\n{ds_path}")
            return

        try:
            import tensorflow as tf
        except ImportError:
            messagebox.showerror("TensorFlow Missing",
                                 "TensorFlow not installed.\nRun: pip install tensorflow")
            return

        try:
            epochs    = int(self._param_vars["epochs"].get())
            batch     = int(self._param_vars["batch"].get())
            img_size  = int(self._param_vars["img_size"].get())
            lr        = float(self._param_vars["lr"].get())
        except ValueError:
            messagebox.showerror("Config Error", "Invalid training parameters.")
            return

        aug       = self._aug_var.get()
        earlystop = self._earlystop_var.get()

        self._training = True
        self._train_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._progress["value"] = 0

        def _train():
            try:
                from tensorflow.keras import layers
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

                self._log("=" * 50)
                self._log("🚀 Starting local training...")
                self._log(f"   Dataset: {ds_path}")
                self._log(f"   Epochs: {epochs} | Batch: {batch} | LR: {lr}")
                self._log(f"   Augmentation: {aug} | Early stop: {earlystop}")
                self._log("=" * 50)

                # Data generators
                if aug:
                    train_gen_obj = ImageDataGenerator(
                        rescale=1./255, rotation_range=25,
                        width_shift_range=0.1, height_shift_range=0.1,
                        zoom_range=0.15, horizontal_flip=True,
                        brightness_range=[0.8,1.2]
                    )
                else:
                    train_gen_obj = ImageDataGenerator(rescale=1./255)

                test_gen_obj = ImageDataGenerator(rescale=1./255)

                train_gen = train_gen_obj.flow_from_directory(
                    str(train_dir), target_size=(img_size,img_size),
                    color_mode="grayscale", batch_size=batch,
                    class_mode="categorical", shuffle=True
                )
                test_gen = test_gen_obj.flow_from_directory(
                    str(test_dir), target_size=(img_size,img_size),
                    color_mode="grayscale", batch_size=batch,
                    class_mode="categorical", shuffle=False
                )
                labels = list(train_gen.class_indices.keys())
                n_cls  = len(labels)
                self._log(f"✅ Classes ({n_cls}): {labels}")

                # Build model
                inp = tf.keras.Input(shape=(img_size,img_size,1))
                def conv_block(x, f, drop=0.25):
                    x = layers.Conv2D(f,(3,3),padding="same",activation="relu")(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Conv2D(f,(3,3),padding="same",activation="relu")(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.MaxPooling2D()(x)
                    x = layers.Dropout(drop)(x)
                    return x

                x = conv_block(inp, 64)
                x = conv_block(x, 128)
                x = conv_block(x, 256)
                x = conv_block(x, 512)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dense(512, activation="relu")(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.5)(x)
                x = layers.Dense(256, activation="relu")(x)
                x = layers.Dropout(0.3)(x)
                out = layers.Dense(n_cls, activation="softmax")(x)
                model = tf.keras.Model(inp, out)

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(lr),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )
                self._log(f"✅ Model built: {model.count_params():,} params")

                # Callbacks
                cbs = [ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0)]
                if earlystop:
                    cbs.append(EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))

                # Custom callback for live updates
                class LiveCallback(tf.keras.callbacks.Callback):
                    def __init__(cb_self):
                        super().__init__()
                    def on_epoch_end(cb_self, epoch, logs=None):
                        if not self._training:
                            cb_self.model.stop_training = True
                        logs = logs or {}
                        pct  = int((epoch+1)/epochs*100)
                        loss = logs.get("loss",0)
                        acc  = logs.get("accuracy",0)
                        vloss= logs.get("val_loss",0)
                        vacc = logs.get("val_accuracy",0)
                        self._log(f"Epoch {epoch+1:>3}/{epochs} | loss:{loss:.4f} acc:{acc:.4f} | val_loss:{vloss:.4f} val_acc:{vacc:.4f}")
                        self.after(0, lambda: self._update_metrics(pct, loss, acc, vloss, vacc, epoch+1, epochs))

                cbs.append(LiveCallback())

                MODEL_DIR.mkdir(exist_ok=True)
                cbs.append(ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor="val_accuracy"))

                # Train
                self._log("🏋 Training started...")
                model.fit(train_gen, validation_data=test_gen, epochs=epochs, callbacks=cbs, verbose=0)

                # Save meta
                meta = {"labels": labels, "val_accuracy": float(model.evaluate(test_gen, verbose=0)[1]),
                        "img_size": img_size}
                with open(META_PATH,"w") as f:
                    json.dump(meta, f, indent=2)

                self._log("=" * 50)
                self._log(f"✅ Training complete! Model saved → {MODEL_PATH}")
                self._log(f"   Val accuracy: {meta['val_accuracy']*100:.2f}%")

                # Reload model in app
                self.after(0, self.app.reload_model)

            except Exception as e:
                import traceback
                self._log(f"❌ Training error:\n{traceback.format_exc()}")
            finally:
                self._training = False
                self.after(0, lambda: self._train_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self._stop_btn.config(state=tk.DISABLED))

        t = threading.Thread(target=_train, daemon=True)
        t.start()
    
    def _stop_training(self):
        self._training = False
        self._log("⏹ Stop requested — will finish current epoch.")
        self._stop_btn.config(state=tk.DISABLED)

    def _update_metrics(self, pct, loss, acc, vloss, vacc, ep, total):
        self._progress["value"] = pct
        self._epoch_lbl.config(text=f"Epoch {ep}/{total} — {pct}%")
        self._met_loss.config(text=f"{loss:.4f}")
        self._met_acc.config(text=f"{acc*100:.1f}%")
        self._met_vlos.config(text=f"{vloss:.4f}")
        self._met_vacc.config(text=f"{vacc*100:.1f}%")

if __name__ == "__main__":
    app = EmotiScanApp()
    app.mainloop()