import os
import sys
import queue
import json
import threading
import time
import datetime  
import asyncio  
import math  
import re  
import tempfile
import io
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk  
from PIL import Image, ImageTk  
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SpkModel  
from groq import Groq  
import edge_tts
import pygame
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

try:
    from ddgs import DDGS  # free DuckDuckGo image search (pip install ddgs)
except ImportError:
    DDGS = None

import glob
import shutil

try:
    import vlc  # in-app video player (needs the 64-bit VLC app installed)
except Exception:
    vlc = None

try:
    import yt_dlp  # finds YouTube videos and their playable streams
except ImportError:
    yt_dlp = None

# yt-dlp needs Deno to unscramble YouTube links. A terminal opened before Deno
# was installed won't have it on PATH, so look in winget's install folders too.
if shutil.which("deno") is None:
    _winget_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet")
    _candidates = [os.path.join(_winget_dir, "Links")] + glob.glob(
        os.path.join(_winget_dir, "Packages", "DenoLand.Deno_*"))
    for _dir in _candidates:
        if os.path.exists(os.path.join(_dir, "deno.exe")):
            os.environ["PATH"] = _dir + os.pathsep + os.environ.get("PATH", "")
            break

VIDEO_HEIGHT = 270  # height of the video panel in pixels

IMAGE_MAX_SIZE = (380, 280)  # largest size an image is shown at in the chat area

MEMORY_MESSAGES = 20          # how many recent messages of the open chat Kooka remembers
MAX_ATTACHMENT_CHARS = 20000  # attached files are trimmed to this many characters
MIC_DEVICE = 1                # microphone device number; the default mic is used if this one fails
SETTINGS_FILE = "kooka_settings.json"  # remembers which chat was open last
GROQ_KEY_FILE = "groq_api_key.txt"     # your private Groq API key (never uploaded to GitHub)


def load_groq_api_key():
    """Read the Groq key from the GROQ_API_KEY environment variable, or from
    GROQ_KEY_FILE next to this program."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), GROQ_KEY_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""

# Chat-app look: colours and sizes
MAIN_BG = "#FFFFFF"      # conversation area (white)
SIDEBAR_BG = "#F9F9F9"   # chat list
HOVER = "#ECECEC"        # hovered / selected items
BUBBLE = "#F1F1F1"       # your message bubbles
COMPOSER_BG = "#F4F4F4"  # the message box
TEXT = "#0D0D0D"
MUTED = "#6B6B6B"
ACCENT = "#10B981"       # Kooka's avatar

# Status colours are chosen for a dark background in many places; on white
# the bright ones are hard to read, so they're swapped for darker shades.
STATUS_ON_LIGHT = {
    "#00E5FF": "#0E7490", "#38BDF8": "#0369A1", "#EAB308": "#A16207",
    "#22C55E": "#15803D", "#F97316": "#C2410C", "#F59E0B": "#B45309",
    "#A855F7": "#7E22CE", "#EF4444": "#DC2626",
}
CHAT_MAX_WIDTH = 760     # the conversation column never gets wider than this
CHAT_SHOW_LIMIT = 60     # messages shown when opening a chat (all are still saved)
ORB_SIZE = 34            # the status orb in the header

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

FAQ_RATE = 16000
BLOCK_SIZE = 4000  
audio_queue = queue.Queue()
STORAGE_FILE = "datalinks.json"
VOICE_PROFILE_FILE = "voice_blueprint.json"

VOICE_ENROLLMENT_PHRASES = [ #for voice recognition
    "Hello, my name is name, and this is my personal voice profile.",
    "Today is a wonderful day to build something extraordinary.",
    "Test, activate your primary systems and establish a connection.",
    "The quick brown fox jumps over the lazy dog.",
    "I authorize this system to recognize my voice as its primary user.",
    "Technology becomes powerful when intelligence meets creativity.",
    "Please check the current time and report the system status.",
    "My voice is unique, and this recording will help identify my speech.",
    "Access granted. All systems are operating within normal parameters.",
    "Test, remember this voice as your authorized voice profile."
]

def audio_callback(indata, frames, time_status, status):
    if status:
        print(status, file=sys.stderr)
    
    audio_chunk = np.frombuffer(indata, dtype=np.int16)
    
    if FAQ_RATE == 16000:
        if frames == 12000 or frames == 4000:
            audio_chunk = audio_chunk[::3]
            
    amplified_chunk = np.multiply(audio_chunk, 3, casting='unsafe')
    cleaned_bytes = np.clip(amplified_chunk, -32768, 32767).astype(np.int16).tobytes()
    audio_queue.put(cleaned_bytes)

class AgentBlueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kooka")
        self.root.geometry("1100x780")  
        self.root.configure(fg_color="#060913") 
        self.root.attributes("-alpha", 1.0)  # solid; slight transparency let other windows show through
        self.root.minsize(820, 640)

        self.root.protocol("WM_DELETE_WINDOW", self.on_secure_close)

        # Thread Data Storage Pools
        self.all_threads = dict()
        self.thread_buttons = dict()
        
        # Guards the saved chats, which are changed from several threads.
        self.data_lock = threading.RLock()

        # Pre-assign standard strings to prevent lifecycle crashes
        self.active_thread_name = "Default_Datalink_01"
        self.chat_history = list()
        
        self.load_datalinks()  
        
        self.session_active = False
        self.last_interaction_time = 0
        self.session_timeout = 15.0  # seconds to carry on talking without the wake word  
        self.selected_file_context = ""
        self.chat_images = []  # keep references so Tk doesn't garbage-collect shown images

        # Stop/Start button state: is a reply playing, and has the user paused it?
        self.is_speaking = False
        self.speech_paused = False

        # VLC video player, created the first time a video is shown.
        self.vlc_instance = None
        self.video_player = None
        self.video_attempt = 0        # bumps for each new video / close, cancelling stale work
        self.video_temp_file = None   # downloaded copy to delete afterwards

        self.owner_voice_vector = self.load_voice_profile()

        # Voice enrollment state. The existing listener remains the only
        # microphone reader, avoiding competing audio streams.
        self.enrollment_active = False
        self.enrollment_waiting = False
        self.enrollment_samples = []
        self.enrollment_sample_event = threading.Event()
        self.enrollment_lock = threading.Lock()

        # Groq API key: kept out of the code (so the code can be shared safely).
        # Put it in groq_api_key.txt next to this file, or set GROQ_API_KEY.
        groq_key = load_groq_api_key()
        if not groq_key:
            self.root.after(500, lambda: self.log_message(
                f"System: No Groq API key found. Put your key in {GROQ_KEY_FILE} next to this "
                "program (or set the GROQ_API_KEY environment variable), then restart Kooka."))
        self.client = Groq(api_key=groq_key or "missing-key")

        # ------------------------------------------------------------------
        # LAYOUT: chat-app style. Chats on the left; on the right a slim header,
        # the conversation in a centred column, and a rounded message box.
        # ------------------------------------------------------------------
        self.root.configure(fg_color=MAIN_BG)

        # ---- Sidebar ----
        self.sidebar_frame = ctk.CTkFrame(root, width=260, corner_radius=0, fg_color=SIDEBAR_BG)
        self.sidebar_frame.pack(side="left", fill="y")
        self.sidebar_frame.pack_propagate(False)

        self.sidebar_title = ctk.CTkLabel(
            self.sidebar_frame, text="Kooka", anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), text_color=TEXT
        )
        self.sidebar_title.pack(fill="x", padx=20, pady=(18, 10))

        self.new_thread_btn = ctk.CTkButton(
            self.sidebar_frame, text="✎   New chat", anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=13), fg_color="transparent",
            hover_color=HOVER, text_color=TEXT, height=38, corner_radius=10,
            command=self.create_new_thread
        )
        self.new_thread_btn.pack(fill="x", padx=10, pady=(0, 12))

        ctk.CTkLabel(
            self.sidebar_frame, text="Chats", anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12), text_color=MUTED
        ).pack(fill="x", padx=20)

        # Voice setup, pinned to the bottom of the sidebar.
        self.voice_box = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.voice_box.pack(side="bottom", fill="x", padx=10, pady=12)
        ctk.CTkFrame(self.voice_box, height=1, fg_color=HOVER).pack(fill="x", pady=(0, 10))
        self.enroll_btn = ctk.CTkButton(
            self.voice_box,
            text="🎙️  Redo my voice setup" if self.owner_voice_vector is not None
            else "🎙️  Set up my voice",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=HOVER, hover_color="#E0E0E0", text_color=TEXT,
            height=34, corner_radius=10, command=self.start_voice_enrollment
        )
        self.enroll_btn.pack(fill="x")
        self.enroll_progress_label = ctk.CTkLabel(
            self.voice_box,
            text="Voice profile: enrolled" if self.owner_voice_vector is not None else "Voice profile: not enrolled",
            font=ctk.CTkFont(family="Segoe UI", size=11), text_color=MUTED,
            anchor="w", justify="left", wraplength=230
        )
        self.enroll_progress_label.pack(fill="x", padx=6, pady=(4, 0))

        self.scroll_frame = ctk.CTkScrollableFrame(
            self.sidebar_frame, fg_color="transparent",
            scrollbar_button_color=SIDEBAR_BG, scrollbar_button_hover_color=HOVER
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=4, pady=4)

        # ---- Main area ----
        self.main_container = ctk.CTkFrame(root, fg_color=MAIN_BG, corner_radius=0)
        self.main_container.pack(side="right", fill="both", expand=True)

        # Header: status orb, name and status on the left; chat name on the right.
        self.header_frame = ctk.CTkFrame(self.main_container, fg_color=MAIN_BG, height=56, corner_radius=0)
        self.header_frame.pack(fill="x", side="top")
        self.header_frame.pack_propagate(False)

        self.center_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")  # holds the orb
        self.center_frame.pack(side="left", padx=(18, 8))
        self.canvas = tk.Canvas(self.center_frame, width=ORB_SIZE, height=ORB_SIZE,
                                bg=MAIN_BG, highlightthickness=0)
        self.canvas.pack()
        orb_c = ORB_SIZE / 2
        self.orb_radius = ORB_SIZE * 0.36
        self.glow_orb = self.canvas.create_oval(
            orb_c - self.orb_radius, orb_c - self.orb_radius,
            orb_c + self.orb_radius, orb_c + self.orb_radius,
            fill="#1E293B", outline="#0EA5E9", width=2
        )
        self.image_path = "image.png"  # add the image you want to display in the centre of the orb
        if os.path.exists(self.image_path):
            pil_img = Image.open(self.image_path).convert("RGBA")
            size = int(ORB_SIZE * 0.55)
            pil_img = pil_img.resize((size, size), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(pil_img)
            self.canvas.create_image(orb_c, orb_c, image=self.tk_image)

        self.title_label = ctk.CTkLabel(
            self.header_frame, text="Kooka",
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"), text_color=TEXT
        )
        self.title_label.pack(side="left")

        self.status_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent",
                                         border_color="#0369A1", border_width=1, corner_radius=12)
        self.status_frame.pack(side="left", padx=14)
        self.status_label = ctk.CTkLabel(
            self.status_frame, text="🎙️ BIOMETRIC ACCESS: ENGAGED", height=22,
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), text_color="#0369A1"
        )
        self.status_label.pack(padx=10, pady=2)

        self.chat_name_label = ctk.CTkLabel(
            self.header_frame, text="",
            font=ctk.CTkFont(family="Segoe UI", size=12), text_color=MUTED
        )
        self.chat_name_label.pack(side="right", padx=20)

        ctk.CTkFrame(self.main_container, height=1, fg_color=HOVER, corner_radius=0).pack(fill="x")

        # ---- Bottom: the message box (packed before the chat so it always keeps its space) ----
        self.bottom_area = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.bottom_area.pack(side="bottom", fill="x", pady=(0, 12))
        self.bottom_column = ctk.CTkFrame(self.bottom_area, fg_color="transparent")
        self.bottom_column.pack(fill="x", padx=24)

        # "Kooka is thinking…" line, just above the message box.
        self.thinking_label = ctk.CTkLabel(
            self.bottom_column, text="", height=20, anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12, slant="italic"), text_color=MUTED
        )
        self.thinking_label.pack(fill="x", padx=12)

        # Attached-file chip; only shown while a file is attached.
        self.file_frame = ctk.CTkFrame(self.bottom_column, fg_color=BUBBLE, corner_radius=12)
        self.file_status_label = ctk.CTkLabel(
            self.file_frame, text="", anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12), text_color=TEXT
        )
        self.file_status_label.pack(side="left", padx=(12, 4), pady=6)
        self.clear_file_btn = ctk.CTkButton(
            self.file_frame, text="✕", width=26, height=26, corner_radius=13,
            fg_color="transparent", hover_color="#DDDDDD", text_color=MUTED,
            command=self.clear_local_file
        )
        self.clear_file_btn.pack(side="left", padx=(0, 6))

        self.input_frame = ctk.CTkFrame(self.bottom_column, fg_color=COMPOSER_BG, corner_radius=26)
        self.input_frame.pack(fill="x")

        self.select_file_btn = ctk.CTkButton(
            self.input_frame, text="＋", width=36, height=36, corner_radius=18,
            font=ctk.CTkFont(size=20), fg_color="transparent", hover_color="#E5E5E5",
            text_color=TEXT, command=self.load_local_file
        )
        self.select_file_btn.pack(side="left", padx=(10, 2), pady=10)

        self.text_input_entry = ctk.CTkEntry(
            self.input_frame, placeholder_text="Message Kooka…",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color=COMPOSER_BG, border_width=0, text_color=TEXT,
            placeholder_text_color=MUTED, height=36
        )
        self.text_input_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.text_input_entry.bind("<Return>", self.process_typed_directive)

        self.send_btn = ctk.CTkButton(
            self.input_frame, text="↑", width=36, height=36, corner_radius=18,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=TEXT, hover_color="#333333", text_color="#FFFFFF",
            command=self.process_typed_directive
        )
        self.send_btn.pack(side="right", padx=(4, 10))

        # Stop/Start as a small icon: ❚❚ pauses Kooka's voice (or a video), ▶ resumes it.
        self.stop_start_btn = ctk.CTkButton(
            self.input_frame, text="❚❚", width=36, height=36, corner_radius=18,
            font=ctk.CTkFont(family="Segoe UI Symbol", size=13),
            fg_color="transparent", hover_color="#E5E5E5", text_color=TEXT,
            command=self.toggle_speech
        )
        self.stop_start_btn.pack(side="right", padx=4)

        self.footer_label = ctk.CTkLabel(
            self.bottom_column, text='Say "Kooka" to talk, or type a message.  Kooka can make mistakes.',
            font=ctk.CTkFont(family="Segoe UI", size=11), text_color=MUTED, height=18
        )
        self.footer_label.pack(pady=(6, 0))

        # Video panel: built now, shown above the chat only while a video is open.
        self.video_panel = ctk.CTkFrame(self.main_container, fg_color="#0B1120", border_color="#1E293B", border_width=1, corner_radius=8)
        self.video_header = ctk.CTkFrame(self.video_panel, fg_color="transparent")
        self.video_header.pack(fill="x", padx=8, pady=(6, 2))
        self.video_title_label = ctk.CTkLabel(
            self.video_header, text="", anchor="w",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), text_color="#38BDF8"
        )
        self.video_title_label.pack(side="left", fill="x", expand=True)
        self.video_close_btn = ctk.CTkButton(
            self.video_header, text="✕ CLOSE VIDEO",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            fg_color="#1E293B", hover_color="#334155", width=110, height=26,
            command=self.close_video
        )
        self.video_close_btn.pack(side="right")
        # VLC draws straight onto this plain Tk frame.
        self.video_surface = tk.Frame(self.video_panel, bg="black", height=VIDEO_HEIGHT)
        self.video_surface.pack(fill="x", padx=8, pady=(2, 8))

        self.terminal_frame = ctk.CTkFrame(self.main_container, fg_color=MAIN_BG, corner_radius=0)
        self.terminal_frame.pack(fill="both", expand=True)

        # The conversation: a scrolling area with a centred column of messages.
        self.chat_scroll = ctk.CTkScrollableFrame(
            self.terminal_frame, fg_color=MAIN_BG, corner_radius=0,
            scrollbar_button_color="#D4D4D4", scrollbar_button_hover_color="#BDBDBD"
        )
        self.chat_scroll.pack(fill="both", expand=True)
        self.chat_column = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        self.chat_column.pack(fill="x", padx=24, pady=12)
        self.wrap_labels = []   # (label, share of column width): re-wrapped when the window resizes
        self.empty_state = None
        self.column_width = 700
        self._last_pad = None
        self.terminal_frame.bind("<Configure>", self._on_chat_resize)

        # Show the chat list and reopen the last chat with its messages.
        self.switch_to_thread(self.active_thread_name)

        pygame.mixer.init()

        if not os.path.exists("model"):
            self.log_message("System Error: 'model' language folder completely missing.\n")
            self.update_status("❌ ENTRY BLOCK REJECTED", "#EF4444", "#EF4444")
            return

        self.model = Model("model")
        
        if os.path.exists("model-spk"):
            self.spk_model = SpkModel("model-spk")
            self.recognizer = KaldiRecognizer(self.model, FAQ_RATE, self.spk_model)
            self.log_message("System: Voice biometric matrix analysis tracking active.")
        else:
            self.recognizer = KaldiRecognizer(self.model, FAQ_RATE)
            self.log_message("System Warning: 'model-spk' directory absent. Running on open acoustic map.")

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.start_async_loop, daemon=True).start()

        self.pulse_phase = 0.0
        self.is_running = True
        # Start the orb animation and microphone listener only once the window
        # is up and running. Starting them earlier could crash both threads
        # ("main thread is not in main loop"), freezing the orb and silently
        # stopping voice input.
        self.root.after(200, self._start_background_threads)

    def _wait_for_window(self):
        """Background threads call this first: wait until the window's main loop
        is really running, so their first update to the screen can't crash."""
        for _ in range(150):  # up to ~30 seconds
            try:
                self.root.after(0, lambda: None)
                return
            except RuntimeError:
                time.sleep(0.2)

    def _start_background_threads(self):
        threading.Thread(target=self.animate_breathing_orb, daemon=True).start()
        self.listen_thread = threading.Thread(target=self.background_listener, daemon=True)
        self.listen_thread.start()

    def start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # ------------------------------------------------------------------
    # PERSISTENCE DATA ARRAYS LOGISTICS
    # ------------------------------------------------------------------
    def load_datalinks(self):
        """Load saved chats. A damaged file is set aside as a backup rather than
        being overwritten (previously a damaged file meant every chat was lost)."""
        data = {}
        if os.path.exists(STORAGE_FILE):
            try:
                with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                backup = f"{STORAGE_FILE}.damaged-{datetime.datetime.now():%Y%m%d-%H%M%S}"
                try:
                    os.replace(STORAGE_FILE, backup)
                    print(f"Saved chats could not be read ({e}); kept a copy as {backup}", file=sys.stderr)
                except OSError:
                    pass
                data = {}

        self.all_threads = {}
        if isinstance(data, dict):
            for name, turns in data.items():
                if isinstance(turns, list):
                    self.all_threads[str(name)] = [
                        t for t in turns if isinstance(t, dict) and "role" in t and "content" in t
                    ]
        if not self.all_threads:
            self.all_threads = {"New chat": []}

        # Reopen the chat that was open last time (otherwise the most recent one).
        last = None
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                last = json.load(f).get("last_chat")
        except Exception:
            pass
        self.active_thread_name = last if last in self.all_threads else list(self.all_threads)[-1]
        self.chat_history = self.all_threads[self.active_thread_name]

    def save_datalinks(self):
        """Save chats safely: write a temporary file and then swap it in, so a
        crash in the middle of saving can't wipe the chat history."""
        with self.data_lock:
            try:
                tmp = STORAGE_FILE + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.all_threads, f, indent=4, ensure_ascii=False)
                os.replace(tmp, STORAGE_FILE)
                with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                    json.dump({"last_chat": self.active_thread_name}, f)
            except Exception as e:
                print(f"Failed to save chats: {e}", file=sys.stderr)

    def load_voice_profile(self):
        if os.path.exists(VOICE_PROFILE_FILE):
            try:
                with open(VOICE_PROFILE_FILE, "r") as f:
                    vector = np.asarray(json.load(f), dtype=np.float32)
                    norm = np.linalg.norm(vector)
                    return vector / norm if norm else None
            except:
                return None
        return None

    def save_voice_profile(self, vector):
        try:
            vector = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("The collected voice profile has a zero-length vector.")
            vector = vector / norm

            with open(VOICE_PROFILE_FILE, "w", encoding="utf-8") as f:
                json.dump(vector.tolist(), f)

            self.owner_voice_vector = vector
            self.log_message("System: Voice profile saved. Speaker matching is enabled.")
            self.root.after(
                0,
                lambda: self.enroll_progress_label.configure(
                    text="Voice profile: enrolled",
                    text_color="#22C55E"
                )
            )
        except Exception as e:
            self.log_message(f"Voice profile could not be saved: {e}")

    def start_voice_enrollment(self):
        if not hasattr(self, "spk_model"):
            self.log_message("Enrollment unavailable: the Vosk 'model-spk' folder is required.")
            self.update_status("⚠️ SPEAKER MODEL REQUIRED", "#F59E0B", "#F59E0B")
            return

        if self.enrollment_active:
            self.log_message("Voice enrollment is already running.")
            return

        with self.enrollment_lock:
            self.enrollment_active = True
            self.enrollment_waiting = False
            self.enrollment_samples = []
            self.enrollment_sample_event.clear()

        self.enroll_btn.configure(state="disabled", text="Voice setup in progress…")
        self.enroll_progress_label.configure(
            text="Preparing voice enrollment...",
            text_color="#F59E0B"
        )
        self.session_active = False
        threading.Thread(target=self._run_voice_enrollment, daemon=True).start()

    def _set_enrollment_progress(self, message, color="#94A3B8"):
        self.root.after(
            0,
            lambda: self.enroll_progress_label.configure(
                text=message,
                text_color=color
            )
        )

    def _run_voice_enrollment(self):
        collected = []
        try:
            self.log_message("System: Voice enrollment started. Please repeat each phrase after it is spoken.")
            for index, phrase in enumerate(VOICE_ENROLLMENT_PHRASES, start=1):
                if not self.is_running:
                    return

                captured = None
                for attempt in range(1, 4):
                    self._set_enrollment_progress(
                        f"Phrase {index}/10 — listen, then repeat (attempt {attempt}/3)",
                        "#38BDF8"
                    )
                    self.speak(f"Phrase {index}. Please repeat: {phrase}")

                    # Ignore audio while TTS is playing. Start accepting the
                    # user's response only after speak() has finished.
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.recognizer.Reset()

                    self.enrollment_sample_event.clear()
                    with self.enrollment_lock:
                        self.enrollment_waiting = True

                    if self.enrollment_sample_event.wait(timeout=25):
                        with self.enrollment_lock:
                            self.enrollment_waiting = False
                            if self.enrollment_samples:
                                captured = self.enrollment_samples[-1]
                    else:
                        with self.enrollment_lock:
                            self.enrollment_waiting = False

                    if captured is not None:
                        break

                    self.log_message(
                        f"System: No usable speaker sample for phrase {index}; please try again."
                    )

                if captured is None:
                    raise TimeoutError(
                        f"Could not collect a speaker sample for phrase {index} after three attempts."
                    )

                collected.append(np.asarray(captured, dtype=np.float32))
                self.log_message(f"System: Voice sample {index}/10 captured.")
                self._set_enrollment_progress(
                    f"Voice samples captured: {index}/10",
                    "#22C55E"
                )

            # Speaker vectors must have matching dimensions before averaging.
            dimensions = {sample.shape for sample in collected}
            if len(dimensions) != 1:
                raise ValueError("The speaker model returned inconsistent sample dimensions.")

            average_vector = np.mean(np.stack(collected, axis=0), axis=0)
            norm = np.linalg.norm(average_vector)
            if norm == 0:
                raise ValueError("The combined voice profile could not be normalized.")

            self.save_voice_profile(average_vector / norm)
            self.log_message("System: All 10 enrollment phrases were collected successfully.")
            self.update_status("🎙️ VOICE PROFILE READY", "#22C55E", "#22C55E")

        except Exception as e:
            self.log_message(f"Voice enrollment stopped: {e}")
            self._set_enrollment_progress("Voice enrollment incomplete — try again", "#F59E0B")
            self.update_status("⚠️ ENROLLMENT INCOMPLETE", "#F59E0B", "#F59E0B")
        finally:
            with self.enrollment_lock:
                self.enrollment_waiting = False
                self.enrollment_active = False
            self.root.after(
                0,
                lambda: self.enroll_btn.configure(
                    state="normal",
                    text="🎙️  Redo my voice setup"
                    if self.owner_voice_vector is not None
                    else "🎙️  Set up my voice"
                )
            )
            self.session_active = False
            if self.is_running:
                self.update_status("🎙️ CORE: STANDBY MODE", "#00E5FF", "#1E293B")

    def on_secure_close(self):
        self.save_datalinks()
        self.is_running = False
        if self.video_player is not None:
            try:
                self.video_player.stop()
            except Exception:
                pass
        self._delete_temp_video()
        self.root.destroy()
        sys.exit(0)

    # ------------------------------------------------------------------
    # DYNAMIC VISUALS
    # ------------------------------------------------------------------
    def animate_breathing_orb(self):
        self._wait_for_window()
        while True:
            if not pygame.mixer.music.get_busy():
                self.pulse_phase += 0.05
                c = ORB_SIZE / 2
                current_radius = ORB_SIZE * 0.36 + 1.5 * math.sin(self.pulse_phase)
                def resize():
                    if hasattr(self, 'canvas') and hasattr(self, 'glow_orb'):
                        self.canvas.coords(self.glow_orb, c-current_radius, c-current_radius, c+current_radius, c+current_radius)
                self.root.after(0, resize)
            time.sleep(0.04)

    def process_typed_directive(self, event=None):
        user_text = self.text_input_entry.get().strip()
        if not user_text:
            return
        self.text_input_entry.delete(0, tk.END)
        self.log_message(f"You: {user_text}")
        self.session_active = False
        threading.Thread(target=self.execute_text_pipeline, args=(user_text,), daemon=True).start()

    def execute_text_pipeline(self, user_text):
        if user_text.lower() in ["bye", "goodbye", "go to sleep", "standby"]:
            self.speak("Understood, sir. Shifting core back to locked standby.")
            self.update_status("🎙️ CORE: STANDBY MODE", "#00E5FF", "#1E293B")
            return
        ai_response = self.ask_groq_brain(user_text)
        self.speak(ai_response)

    # ------------------------------------------------------------------
    # CHATS (sidebar): new, switch, rename, delete
    # ------------------------------------------------------------------
    def _unique_chat_name(self, base, ignore=None):
        name, n = base, 2
        while name in self.all_threads and name != ignore:
            name = f"{base} {n}"
            n += 1
        return name

    def create_new_thread(self):
        # Don't pile up blank chats: if this one is still empty, just use it.
        if not self.chat_history:
            self.log_message("System: This chat is already empty — just start talking or typing.")
            return
        name = self._unique_chat_name("New chat")
        with self.data_lock:
            self.all_threads[name] = []
        self.switch_to_thread(name)

    def switch_to_thread(self, name):
        if name not in self.all_threads:
            return
        self.active_thread_name = name
        self.chat_history = self.all_threads[name]
        self._clear_chat_view()
        if not self.chat_history:
            self._show_empty_state()
        shown = self.chat_history[-CHAT_SHOW_LIMIT:]
        if len(shown) < len(self.chat_history):
            self.log_message(f"System: Showing the latest {len(shown)} of {len(self.chat_history)} messages.")
        for turn in shown:
            who = "You" if turn["role"] == "user" else "Kooka"
            self.log_message(f"{who}: {turn['content']}", stamp=turn.get("time"))
        self.refresh_sidebar_ui()
        self.save_datalinks()  # remembers this as the chat to reopen next time

    def refresh_sidebar_ui(self):
        for row in self.thread_buttons.values():
            row.destroy()
        self.thread_buttons.clear()
        # Most recently used chats at the top.
        for name in reversed(list(self.all_threads.keys())):
            is_active = (name == self.active_thread_name)
            row = ctk.CTkFrame(self.scroll_frame, corner_radius=10,
                               fg_color=HOVER if is_active else "transparent")
            row.pack(fill="x", padx=4, pady=2)
            label = name if len(name) <= 22 else name[:21] + "…"
            ctk.CTkButton(
                row, text=label, anchor="w", height=34, corner_radius=10,
                font=ctk.CTkFont(family="Segoe UI", size=13),
                fg_color="transparent", hover_color=HOVER,
                text_color=TEXT if is_active else "#4A4A4A",
                command=lambda n=name: self.switch_to_thread(n)
            ).pack(side="left", fill="x", expand=True)
            more_btn = ctk.CTkButton(
                row, text="⋯", width=26, height=32, fg_color="transparent",
                hover_color="#E0E0E0", text_color=MUTED,
                font=ctk.CTkFont(size=14, weight="bold")
            )
            more_btn.configure(command=lambda n=name, w=more_btn: self._chat_menu(n, w))
            more_btn.pack(side="right", padx=(0, 2))
            self.thread_buttons[name] = row
        self.chat_name_label.configure(text=f"💬 {self.active_thread_name}")

    def _chat_menu(self, name, widget):
        """The ⋯ button next to each chat: rename or delete it."""
        menu = tk.Menu(self.root, tearoff=0, bg=COMPOSER_BG, fg=TEXT,
                       activebackground="#E5E5E5", activeforeground=TEXT, bd=0)
        menu.add_command(label="✎  Rename", command=lambda: self.rename_chat(name))
        menu.add_command(label="🗑  Delete", command=lambda: self.delete_chat(name))
        menu.tk_popup(widget.winfo_rootx(), widget.winfo_rooty() + widget.winfo_height())

    def rename_chat(self, name):
        dialog = ctk.CTkInputDialog(text=f"New name for “{name}”:", title="Rename chat")
        new_name = (dialog.get_input() or "").strip()[:40]
        if not new_name or new_name == name or name not in self.all_threads:
            return
        with self.data_lock:
            new_name = self._unique_chat_name(new_name, ignore=name)
            # Rebuild the dict so the chat keeps its place in the list.
            self.all_threads = {(new_name if k == name else k): v for k, v in self.all_threads.items()}
            if self.active_thread_name == name:
                self.active_thread_name = new_name
        self.refresh_sidebar_ui()
        self.save_datalinks()

    def delete_chat(self, name):
        if not messagebox.askyesno("Delete chat", f"Delete “{name}”?\nThis can't be undone.", parent=self.root):
            return
        with self.data_lock:
            self.all_threads.pop(name, None)
            if not self.all_threads:
                self.all_threads["New chat"] = []
        if name == self.active_thread_name:
            self.switch_to_thread(list(self.all_threads)[-1])
        else:
            self.refresh_sidebar_ui()
            self.save_datalinks()

    def _record_exchange(self, history, user_input, reply):
        """Save a question and answer to the chat they belong to, move that chat
        to the top, and name a 'New chat' after its first question."""
        now = datetime.datetime.now().isoformat(timespec="seconds")
        with self.data_lock:
            first_message = not history
            history.append({"role": "user", "content": user_input, "time": now})
            history.append({"role": "assistant", "content": reply, "time": now})
            name = next((k for k, v in self.all_threads.items() if v is history), None)
            if name is not None:
                new_name = name
                if first_message and (name.startswith("New chat") or name.startswith("Datalink_Stream_")
                                      or name == "Default_Datalink_01"):
                    title = re.sub(r"\s+", " ", user_input).strip()
                    title = title[:1].upper() + title[1:]
                    if len(title) > 32:
                        title = title[:31].rstrip() + "…"
                    new_name = self._unique_chat_name(title or name, ignore=name)
                # Re-adding puts it last in the dict, which is the top of the sidebar.
                self.all_threads.pop(name)
                self.all_threads[new_name] = history
                if self.active_thread_name == name:
                    self.active_thread_name = new_name
        self.save_datalinks()
        self.root.after(0, self.refresh_sidebar_ui)

    def load_local_file(self):
        file_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Select Context Payload", filetypes=(("Text Files", "*.txt"), ("Log Files", "*.log"), ("All Files", "*.*")))
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError as e:
            self.log_message(f"System: Couldn't open that file: {e}")
            return
        note = ""
        if len(content) > MAX_ATTACHMENT_CHARS:
            # Very large files would be too big to send to the AI with every message.
            content = content[:MAX_ATTACHMENT_CHARS]
            note = "  (large file: only the start is used)"
        self.selected_file_context = content
        self.file_status_label.configure(text=f"📄 {os.path.basename(file_path)}{note}")
        if not self.file_frame.winfo_ismapped():
            self.file_frame.pack(anchor="w", pady=(0, 6), before=self.input_frame)

    def clear_local_file(self):
        self.selected_file_context = ""
        self.file_frame.pack_forget()

    def update_status(self, text, text_color, orb_color=None):
        text_color = STATUS_ON_LIGHT.get(text_color, text_color)
        def update():
            self.status_label.configure(text=text, text_color=text_color)
            self.status_frame.configure(border_color=text_color)
            if orb_color:
                self.canvas.itemconfig(self.glow_orb, fill=orb_color, outline=orb_color)
        self.root.after(0, update)

    def _format_stamp(self, iso=None):
        """'14:32' for today, '09 Nov 14:32' for older messages."""
        try:
            when = datetime.datetime.fromisoformat(iso) if iso else datetime.datetime.now()
        except (TypeError, ValueError):
            return ""
        return when.strftime("%H:%M" if when.date() == datetime.date.today() else "%d %b %H:%M")

    def log_message(self, text, kind=None, stamp=None):
        """Add a line to the chat. 'You…:' and 'Kooka:' lines get a coloured name,
        a time and an indented message; anything else is a small grey system note."""
        if kind is None:
            if text.startswith("You"):
                kind = "user"
            elif text.startswith("Kooka:"):
                kind = "kooka"
            else:
                kind = "system"
        who, body = "", ""
        if kind in ("user", "kooka"):
            if ":" in text:
                who, _, body = text.partition(":")
            else:
                kind = "system"
        stamp_text = self._format_stamp(stamp)

        self.root.after(0, lambda: self._add_message_widget(kind, who, body, text, stamp_text))

    # ---- chat view: message bubbles in a centred column ----
    def _add_message_widget(self, kind, who, body, text, stamp_text):
        col = self.chat_column
        if kind == "user":
            self._clear_empty_state()
            row = ctk.CTkFrame(col, fg_color="transparent")
            row.pack(fill="x", pady=(14, 0))
            bubble = ctk.CTkFrame(row, fg_color=BUBBLE, corner_radius=18)
            bubble.pack(anchor="e")
            label = ctk.CTkLabel(bubble, text=body.strip(), justify="left", anchor="w",
                                 font=("Segoe UI", 14), text_color=TEXT)
            label.pack(padx=16, pady=9)
            self._track_wrap(label, 0.72)
            if stamp_text:
                ctk.CTkLabel(row, text=stamp_text, height=16, font=("Segoe UI", 10),
                             text_color=MUTED).pack(anchor="e", padx=8)
        elif kind == "kooka":
            self._clear_empty_state()
            row = ctk.CTkFrame(col, fg_color="transparent")
            row.pack(fill="x", pady=(14, 0))
            head = ctk.CTkFrame(row, fg_color="transparent")
            head.pack(anchor="w")
            ctk.CTkLabel(head, text="K", width=26, height=26, corner_radius=13, fg_color=ACCENT,
                         text_color="#FFFFFF", font=("Segoe UI", 12, "bold")).pack(side="left")
            ctk.CTkLabel(head, text=f"  {who.strip()}", font=("Segoe UI", 13, "bold"),
                         text_color=TEXT).pack(side="left")
            if stamp_text:
                ctk.CTkLabel(head, text=f"  {stamp_text}", font=("Segoe UI", 10),
                             text_color=MUTED).pack(side="left")
            label = ctk.CTkLabel(row, text=body.strip(), justify="left", anchor="w",
                                 font=("Segoe UI", 14), text_color=TEXT)
            label.pack(anchor="w", padx=(34, 0), pady=(2, 0))
            self._track_wrap(label, 0.92)
        else:
            note = text.strip()
            if note.startswith("System:"):
                note = note[len("System:"):].strip()
            label = ctk.CTkLabel(col, text=note, justify="left", anchor="w",
                                 font=("Segoe UI", 11), text_color=MUTED)
            label.pack(anchor="w", padx=(34, 0), pady=(3, 0))
            self._track_wrap(label, 0.92)
        self._make_copyable(label, body.strip() if kind in ("user", "kooka") else text.strip())
        self._scroll_to_bottom()

    def _track_wrap(self, label, share):
        """Wrap a message to the column width, and keep it wrapped when resizing."""
        label.configure(wraplength=int(self.column_width * share))
        self.wrap_labels.append((label, share))

    def _make_copyable(self, label, value):
        """Right-click a message to copy its text."""
        def copy(_event=None):
            self.root.clipboard_clear()
            self.root.clipboard_append(value)
            self.update_status("📋 COPIED TO CLIPBOARD", "#38BDF8")
        label.bind("<Button-3>", copy)

    def _scroll_to_bottom(self):
        def go():
            canvas = self.chat_scroll._parent_canvas
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))  # height of what's really there
            canvas.yview_moveto(1.0)
        self.root.after(60, go)

    def _on_chat_resize(self, event=None):
        """Keep the conversation (and the message box) in a centred column."""
        width = self.terminal_frame.winfo_width()
        col = min(CHAT_MAX_WIDTH, max(320, width - 64))
        pad = max(16, (width - col) // 2)
        if col == self.column_width and pad == self._last_pad:
            return
        self.column_width, self._last_pad = col, pad
        self.chat_column.pack_configure(padx=max(8, pad - 10))  # the scrollbar takes ~10px
        self.bottom_column.pack_configure(padx=pad)
        for label, share in self.wrap_labels:
            try:
                label.configure(wraplength=int(col * share))
            except tk.TclError:
                pass  # message was removed

    def _clear_chat_view(self):
        # Replace the whole column with a fresh one. (Just emptying it left Tk
        # thinking it was still as tall as the previous chat, so it scrolled
        # down into blank space.)
        self.chat_column.destroy()
        self.chat_column = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        self.chat_column.pack(fill="x", padx=max(8, (self._last_pad or 34) - 10), pady=12)
        self.chat_scroll._parent_canvas.yview_moveto(0)
        self.wrap_labels = []
        self.chat_images = []
        self.empty_state = None

    def _show_empty_state(self):
        self._clear_empty_state()
        self.empty_state = ctk.CTkFrame(self.chat_column, fg_color="transparent")
        self.empty_state.pack(fill="x", pady=(150, 20))
        ctk.CTkLabel(self.empty_state, text="What can I help with?",
                     font=("Segoe UI", 28, "bold"), text_color=TEXT).pack()
        ctk.CTkLabel(self.empty_state, text='Type below, or say "Kooka" and ask out loud.',
                     font=("Segoe UI", 13), text_color=MUTED).pack(pady=(8, 0))

    def _clear_empty_state(self):
        if self.empty_state is not None:
            self.empty_state.destroy()
            self.empty_state = None

    def _set_thinking(self, on):
        self.root.after(0, lambda: self.thinking_label.configure(
            text="Kooka is thinking…" if on else ""))

    async def _async_speak(self, text):
        """Speak the whole reply as one continuous clip (no gaps between sentences)."""
        if not hasattr(self, "speech_lock"):
            self.speech_lock = asyncio.Lock()

        async with self.speech_lock:
            clean_text = re.sub(r"\[.*?\]|\(.*?\)", "", str(text))
            clean_text = clean_text.replace("*", "").strip()
            if not clean_text:
                clean_text = "Datalink operational."

            voice = "en-GB-RyanNeural" #edge tts voice, you can change it to any other available voice
            rate = "+12%"

            try:
                # One streamed request for the whole reply: a single connection
                # (~0.8s total) instead of ~1.2s per sentence, and natural flow
                # between sentences because it is synthesized as one utterance.
                audio = bytearray()
                for attempt in range(2):
                    audio.clear()
                    try:
                        communicate = edge_tts.Communicate(clean_text, voice, rate=rate)
                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                audio.extend(chunk["data"])
                        if audio:
                            break
                    except Exception:
                        if attempt == 1:
                            raise
                if not audio:
                    raise RuntimeError("Ryan Neural returned no audio.")

                # Play straight from memory: no temporary files to write or delete.
                pygame.mixer.music.load(io.BytesIO(bytes(audio)), "mp3")
                self.speech_paused = False
                pygame.mixer.music.play()
                self.is_speaking = True
                # get_busy() is False while paused, so keep waiting until the
                # user presses Start again instead of treating it as finished.
                while pygame.mixer.music.get_busy() or self.speech_paused:
                    await asyncio.sleep(0.02)
            except Exception as exc:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
                error_text = f"TTS playback error: {type(exc).__name__}: {exc}"
                print(error_text, file=sys.stderr)
                self.log_message("System: " + error_text)
            finally:
                self.is_speaking = False
                self.speech_paused = False
                self.root.after(0, self._refresh_stop_start_color)
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass

    def _refresh_stop_start_color(self):
        """Show ▶ when something is paused (press to continue), ❚❚ otherwise."""
        if self.speech_paused or self._video_state() == "paused":
            self.stop_start_btn.configure(text="▶", text_color=ACCENT)
        else:
            self.stop_start_btn.configure(text="❚❚", text_color=TEXT)

    def toggle_speech(self):
        """Stop/Start button: pause or resume Kooka's voice, or else the video."""
        if self.speech_paused:
            pygame.mixer.music.unpause()
            self.speech_paused = False
            self.update_status("🔊 AUDIO: TRANSMITTING...", "#22C55E", "#22C55E")
        elif self.is_speaking and pygame.mixer.music.get_busy():
            self.speech_paused = True
            pygame.mixer.music.pause()
            self.update_status("⏸ AUDIO: PAUSED", "#F59E0B", "#F59E0B")
        elif self._video_state() == "playing":
            self.video_player.set_pause(1)
            self.update_status("⏸ VIDEO: PAUSED", "#F59E0B", "#F59E0B")
        elif self._video_state() == "paused":
            self.video_player.set_pause(0)
            self.update_status("🎬 VIDEO: PLAYING", "#A855F7", "#A855F7")
        else:
            self.log_message("System: Nothing is playing right now.")
        self._refresh_stop_start_color()

    # ------------------------------------------------------------------
    # INTERNET VIDEOS (YouTube via yt-dlp, played in-app with VLC)
    # ------------------------------------------------------------------
    def _video_state(self):
        """'playing', 'paused', or None when no video is active."""
        if self.video_player is None or vlc is None:
            return None
        state = self.video_player.get_state()
        if state in (vlc.State.Opening, vlc.State.Buffering, vlc.State.Playing):
            return "playing"
        if state == vlc.State.Paused:
            return "paused"
        return None

    def fetch_and_play_video(self, query):
        if vlc is None or yt_dlp is None:
            self.log_message("System: Video playback needs VLC plus the python-vlc and yt-dlp packages.")
            return
        self.video_attempt += 1
        attempt = self.video_attempt
        self.log_message(f"System: Searching YouTube for a video of '{query}'...")
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                                   "extract_flat": True, "noplaylist": True}) as ydl:
                found = ydl.extract_info(f"ytsearch5:{query}", download=False)
            entries = [e for e in (found.get("entries") or [])
                       if (e.get("url") or e.get("id")) and e.get("live_status") != "is_live"]
        except Exception as e:
            self.log_message(f"System: Video search failed: {e}")
            return

        # Prefer one H.264 stream with sound built in (plays everywhere); otherwise
        # take separate video and audio streams and let VLC play them together.
        # AV1 is a last resort because VLC often shows it as a black picture.
        stream_opts = {
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "format": "b[height<=720][acodec!=none][vcodec^=avc1]"
                      "/b[height<=720][acodec!=none][vcodec!=none]"
                      "/bv*[height<=720][vcodec^=avc1]+ba"
                      "/bv*[height<=720]+ba/b",
        }
        for entry in entries:
            page_url = entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}"
            try:
                with yt_dlp.YoutubeDL(stream_opts) as ydl:
                    info = ydl.extract_info(page_url, download=False)
            except Exception as e:
                print(f"Video stream lookup failed for {page_url}: {e}", file=sys.stderr)
                continue
            parts = info.get("requested_formats") or [info]
            video_url = parts[0].get("url")
            audio_url = parts[1].get("url") if len(parts) > 1 else None
            if not video_url:
                continue
            user_agent = (info.get("http_headers") or {}).get("User-Agent")
            title = info.get("title") or entry.get("title") or query

            # Let Kooka finish talking before the video starts.
            waited = 0.0
            while (self.is_speaking or self.speech_paused) and waited < 300:
                time.sleep(0.2)
                waited += 0.2
            self.root.after(0, lambda: self._play_video(
                video_url, audio_url, user_agent, title, page_url, attempt))
            return
        self.log_message(f"System: Couldn't find a playable video of '{query}'.")

    def _check_video_started(self, page_url, title, attempt, checks=0):
        """YouTube sometimes refuses to let VLC stream a video, leaving a black
        panel. If nothing has played after a few seconds, download it instead."""
        if attempt != self.video_attempt or self.video_player is None:
            return
        state = self.video_player.get_state()
        if self.video_player.get_time() > 0 or state == vlc.State.Paused:
            return  # it's playing fine
        refused = state in (vlc.State.Ended, vlc.State.Error, vlc.State.Stopped)
        if refused or checks >= 5:  # refused, or still stuck loading after ~15s
            self.video_player.stop()
            threading.Thread(target=self._download_and_play,
                             args=(page_url, title, attempt), daemon=True).start()
        else:
            self.root.after(3000, lambda: self._check_video_started(page_url, title, attempt, checks + 1))

    def _download_and_play(self, page_url, title, attempt):
        self.log_message("System: YouTube refused direct streaming, so I'm downloading the video instead...")
        folder = os.path.join(tempfile.gettempdir(), "kooka_videos")
        os.makedirs(folder, exist_ok=True)
        last_shown = [-10]

        def progress(d):
            if d.get("status") != "downloading" or attempt != self.video_attempt:
                return
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                percent = int(d.get("downloaded_bytes", 0) * 100 / total)
                if percent >= last_shown[0] + 10:
                    last_shown[0] = percent
                    self.update_status(f"⬇ VIDEO: DOWNLOADING {percent}%", "#A855F7", "#A855F7")

        opts = {
            "quiet": True, "no_warnings": True, "noplaylist": True, "overwrites": True,
            "noprogress": True,  # progress goes to the status bar instead of the terminal
            # A single file with sound built in, so no extra merging tool is needed.
            "format": "b[height<=480][acodec!=none][vcodec^=avc1]"
                      "/b[height<=480][acodec!=none][vcodec!=none]"
                      "/b[acodec!=none][vcodec!=none]",
            "outtmpl": os.path.join(folder, "%(id)s.%(ext)s"),
            "progress_hooks": [progress],
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(page_url, download=True)
                path = ydl.prepare_filename(info)
        except Exception as e:
            self.log_message(f"System: Couldn't download the video either: {e}")
            self.update_status("⚠️ VIDEO UNAVAILABLE", "#EF4444", "#EF4444")
            return
        if attempt != self.video_attempt:  # closed, or another video was asked for
            try:
                os.remove(path)
            except OSError:
                pass
            return
        self.root.after(0, lambda: self._play_video(path, None, None, title, None, attempt, temp_file=path))

    def _delete_temp_video(self):
        if self.video_temp_file:
            try:
                os.remove(self.video_temp_file)
            except OSError:
                pass
            self.video_temp_file = None

    def _play_video(self, video_url, audio_url, user_agent, title, page_url, attempt, temp_file=None):
        if attempt != self.video_attempt:
            return  # the video was closed or replaced while it was loading
        try:
            if self.vlc_instance is None:
                self.vlc_instance = vlc.Instance("--no-video-title-show", "--quiet")
                self.video_player = self.vlc_instance.media_player_new()
            self.video_player.stop()
            if self.video_temp_file != temp_file:
                self._delete_temp_video()
            self.video_temp_file = temp_file

            # Show the panel above the conversation before handing it to VLC.
            self.video_title_label.configure(text=f"🎬 {title[:70]}")
            if not self.video_panel.winfo_ismapped():
                self.video_panel.pack(padx=25, pady=5, fill="x", before=self.terminal_frame)
            self.root.update_idletasks()
            self.video_player.set_hwnd(self.video_surface.winfo_id())

            media = self.vlc_instance.media_new(video_url)
            if audio_url:
                media.add_option(f":input-slave={audio_url}")
            if user_agent:
                media.add_option(f":http-user-agent={user_agent}")
            if page_url:
                media.add_option(":http-referrer=https://www.youtube.com/")
            self.video_player.set_media(media)
            self.video_player.play()
            self.log_message(f"System: Playing video: {title}")
            self.update_status("🎬 VIDEO: PLAYING", "#A855F7", "#A855F7")
            self._refresh_stop_start_color()
            if page_url:  # streaming: make sure it actually starts
                self.root.after(3000, lambda: self._check_video_started(page_url, title, attempt))
        except Exception as e:
            self.log_message(f"System: Couldn't play the video: {e}")

    def close_video(self):
        self.video_attempt += 1  # cancels any lookup or download still running
        if self.video_player is not None:
            self.video_player.stop()
        self._delete_temp_video()
        self.video_panel.pack_forget()
        self._refresh_stop_start_color()
        self.update_status("🎙️ CORE: STANDBY MODE", "#00E5FF", "#1E293B")

    def speak(self, text):
        # Don't talk over a video: pause it (Stop/Start resumes it afterwards).
        if self._video_state() == "playing":
            self.video_player.set_pause(1)
        self.update_status("🔊 AUDIO: TRANSMITTING...", "#22C55E", "#22C55E")
        self.log_message(f"Kooka: {text}")
        self._set_thinking(False)

        try:
            future = asyncio.run_coroutine_threadsafe(self._async_speak(text), self.loop)
            # Surface unexpected coroutine/loop failures instead of silently skipping audio.
            # The 3-minute limit restarts while paused, so a long pause isn't reported as a failure.
            deadline = time.time() + 180
            while True:
                try:
                    future.result(timeout=0.5)
                    break
                except TimeoutError:
                    if self.speech_paused:
                        deadline = time.time() + 180
                    elif time.time() > deadline:
                        raise
        except Exception as e:
            error_text = f"Speech request failed: {type(e).__name__}: {e}"
            print(error_text, file=sys.stderr)
            self.log_message("System: " + error_text)
            self.update_status("⚠️ AUDIO PLAYBACK ERROR", "#EF4444", "#EF4444")
            return

        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            self.recognizer.Reset()
        except Exception as e:
            self.log_message(f"System: Recognizer reset warning: {e}")

        self.session_active = True
        self.last_interaction_time = time.time()
        self.update_status("👀 SESSION OPEN (FEEDBACK)...", "#F97316", "#F97316")

    def ask_groq_brain(self, user_input):
        self.update_status("🧠 SYNAPSE: COMPUTING...", "#EAB308", "#EAB308") 
        self._set_thinking(True)

        # Remember which chat this question belongs to, in case you switch chats
        # while Kooka is still answering.
        history = self.chat_history

        # Answer clock questions instantly. Only real time questions count now;
        # before, any message containing "time" (like "sometimes") got the clock.
        if re.search(r"\bwhat(?:'s| is)? the time\b|\bwhat time is it\b|\bcurrent time\b|"
                     r"\btell me the time\b|\btime is it\b",
                     user_input, flags=re.IGNORECASE):
            reply = f"The current time is {datetime.datetime.now().strftime('%I:%M %p').lstrip('0')}."
            self._record_exchange(history, user_input, reply)
            return reply

        system_prompt = (
            "Your name is Kooka. You are a helpful personal voice assistant. "
            "If anyone asks your name or who you are, say you are Kooka. "
            "Never say you are ChatGPT or mention OpenAI. "
            "Keep answers short and conversational, since they are spoken aloud. "
            "You have a web search tool: use it for anything current or time-sensitive "
            "(news, weather, sports, prices, recent events) and answer from the results. "
            "Do not include links, citations, or source markers in your answer. "
            "You can show pictures in the chat window. When the user asks to see, show, or "
            "get a picture/photo/image of something, or when a picture would clearly help, "
            "add a tag like [IMAGE: short search terms] at the end of your reply "
            "(at most 2 tags), and say briefly in words what you are showing. "
            "Only if the user's LATEST message names a specific site, add it after a pipe: "
            "[IMAGE: cozy bedroom ideas | pinterest]. Supported sites: pinterest, flickr, wikipedia. "
            "Otherwise do not add a site, even if an earlier message mentioned one. "
            "Never claim you cannot show images. "
            "You can also play videos in the app. When the user asks to watch, see, or play "
            "a video, clip, song, or trailer, add a tag like [VIDEO: short search terms] at the "
            "end of your reply (only 1 video tag), and say briefly what you are playing. "
            "Never claim you cannot play videos. "
            f"Today's date is {datetime.datetime.now().strftime('%A, %d %B %Y')}."
        )
            #add your system prompt to make it act how you want
        
        if self.selected_file_context:
            system_prompt += f"\nThe user attached this file; use it when relevant:\n{self.selected_file_context}"

        messages = [{"role": "system", "content": system_prompt}]
        # Kooka now remembers the last MEMORY_MESSAGES messages (was 4). Only role
        # and text are sent; the saved times aren't part of the AI's format.
        for msg in history[-MEMORY_MESSAGES:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        try:
            # browser_search lets the model look things up on the internet when it
            # decides a question needs live information. Searching uses extra tokens,
            # so the limit is higher; the system prompt keeps spoken answers short.
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                tools=[{"type": "browser_search"}],
                max_tokens=1024,
            )
            reply = response.choices[0].message.content or ""
            reply = re.sub(r'【[^】]*】', '', reply)          # strip search citation markers
            reply = re.sub(r'https?://\S+', '', reply)       # never read URLs aloud

            # Pull out [IMAGE: ...] tags, fetch those pictures in the background,
            # and keep the tags out of the spoken/displayed text.
            image_queries = re.findall(r'\[\s*IMAGE\s*:\s*([^\]]+)\]', reply, flags=re.IGNORECASE)
            reply = re.sub(r'\[\s*IMAGE\s*:[^\]]*\]', '', reply, flags=re.IGNORECASE)
            for query in image_queries[:2]:
                threading.Thread(target=self.fetch_and_show_image, args=(query.strip(),), daemon=True).start()

            # Same for [VIDEO: ...] tags: find one video and play it in the app.
            video_queries = re.findall(r'\[\s*VIDEO\s*:\s*([^\]]+)\]', reply, flags=re.IGNORECASE)
            reply = re.sub(r'\[\s*VIDEO\s*:[^\]]*\]', '', reply, flags=re.IGNORECASE)
            if video_queries:
                threading.Thread(target=self.fetch_and_play_video, args=(video_queries[0].strip(),), daemon=True).start()

            reply = re.sub(r'\*.*?\*', '', reply).replace("*", "").strip()
            if not reply and video_queries: reply = "Here's a video."
            if not reply and image_queries: reply = "Here's what I found."
            if not reply: reply = "Datalink stream stabilized. Awaiting your commands, Sir."
            
            self._record_exchange(history, user_input, reply)
            return reply
        except Exception as e:
            # Show the details in the chat, but only speak a short message
            # (it used to read the whole technical error aloud).
            print(f"AI request failed: {e}", file=sys.stderr)
            self.log_message(f"System: AI request failed — {type(e).__name__}: {str(e)[:200]}")
            return "Sorry, I couldn't reach my brain just now. Please try again in a moment."

    # ------------------------------------------------------------------
    # INTERNET IMAGES
    # ------------------------------------------------------------------
    # Each source returns a list of (image_url, credit_text) candidates.
    def _images_from_duckduckgo(self, query):
        if DDGS is None:
            return []
        results = DDGS().images(query, safesearch="on", max_results=6)
        return [(r["image"], "DuckDuckGo") for r in results if r.get("image")]

    def _images_from_pinterest(self, query):
        # Pinterest has no public search API, so ask the search engine for pins
        # and keep only images actually hosted on Pinterest's image servers.
        # DuckDuckGo's engine ignores "site:", so use engines that honour it and
        # move on to the next if one returns no pins or fails.
        if DDGS is None:
            return []
        for backend in ("bing", "yahoo", "auto"):
            try:
                results = DDGS().images(f"{query} site:pinterest.com", safesearch="on",
                                        max_results=12, backend=backend)
            except Exception as e:
                print(f"Pinterest search via {backend} failed: {e}", file=sys.stderr)
                continue
            out = []
            for r in results:
                url = r.get("image", "")
                if "pinimg.com" in url:
                    title = (r.get("title") or "").split("|")[0].strip()
                    out.append((url, f"Pinterest · {title[:50]}" if title else "Pinterest"))
            if out:
                return out
        return []

    def _images_from_openverse(self, query):
        # Openverse: free, no key; openly licensed images from Flickr, Wikimedia, museums...
        data = requests.get(
            "https://api.openverse.org/v1/images/",
            params={"q": query, "page_size": 5, "mature": "false"},
            headers={"User-Agent": "KookaAssistant/1.0"}, timeout=8,
        ).json()
        out = []
        for item in data.get("results", []):
            source = (item.get("source") or "Openverse").replace("_", " ").title()
            credit = f"{source} via Openverse"
            if item.get("creator"):
                credit += f" · by {item['creator']}"
            if item.get("license"):
                credit += f" (CC {item['license'].upper()})"
            for url in (item.get("thumbnail"), item.get("url")):
                if url:
                    out.append((url, credit))
        return out

    def _images_from_wikipedia(self, query):
        data = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=dict(action="query", format="json", generator="search",
                        gsrsearch=query, gsrlimit=1, prop="pageimages",
                        piprop="thumbnail", pithumbsize=600),
            headers={"User-Agent": "KookaAssistant/1.0"}, timeout=6,
        ).json()
        return [(p["thumbnail"]["source"], "Wikipedia")
                for p in data.get("query", {}).get("pages", {}).values()
                if p.get("thumbnail", {}).get("source")]

    def _search_images(self, query, preferred_source=None):
        """Search every source at the same time; return candidates in priority order.
        If preferred_source is given (e.g. 'pinterest'), that source is tried first."""
        sources = [
            ("DuckDuckGo", self._images_from_duckduckgo),
            ("Pinterest", self._images_from_pinterest),
            ("Openverse", self._images_from_openverse),
            ("Wikipedia", self._images_from_wikipedia),
        ]
        if preferred_source:
            # "flickr" is served through Openverse
            wanted = {"flickr": "openverse"}.get(preferred_source.lower(), preferred_source.lower())
            sources.sort(key=lambda s: s[0].lower() != wanted)
        with ThreadPoolExecutor(max_workers=len(sources)) as pool:
            futures = [(name, pool.submit(fn, query)) for name, fn in sources]
            candidates = []
            for name, future in futures:
                try:
                    candidates += future.result(timeout=15)
                except Exception as e:
                    print(f"{name} image search failed: {e}", file=sys.stderr)
        return candidates

    def fetch_and_show_image(self, query):
        # Tags may name a source: "[IMAGE: cozy bedroom | pinterest]"
        query, _, preferred = (part.strip() for part in query.partition("|"))
        where = f" on {preferred.title()}" if preferred else ""
        self.log_message(f"System: Searching{where} for an image of '{query}'...")
        candidates = self._search_images(query, preferred or None)
        if preferred:
            source_word = {"flickr": "openverse"}.get(preferred.lower(), preferred.lower())
            if not any(source_word in credit.lower() for _, credit in candidates):
                self.log_message(f"System: Nothing found on {preferred.title()}; showing a picture from another source.")
        for url, credit in candidates:
            try:
                resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
                img.load()
                img = img.convert("RGB")
                img.thumbnail(IMAGE_MAX_SIZE, Image.Resampling.LANCZOS)
            except Exception:
                continue  # broken link or not an image: try the next result
            self.root.after(0, lambda i=img, q=f"{query} — {credit}": self._insert_image(i, q))
            return
        self.log_message(f"System: Couldn't find a picture of '{query}'.")

    def _insert_image(self, pil_img, caption):
        self._clear_empty_state()
        photo = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        self.chat_images.append(photo)  # keep a reference so it isn't garbage-collected
        row = ctk.CTkFrame(self.chat_column, fg_color="transparent")
        row.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(row, image=photo, text="").pack(anchor="w", padx=(34, 0))
        ctk.CTkLabel(row, text=f"🖼 {caption}", font=("Segoe UI", 11), text_color=MUTED,
                     anchor="w").pack(anchor="w", padx=(34, 0), pady=(2, 0))
        self._scroll_to_bottom()

    def _open_microphone(self):
        """Open MIC_DEVICE; if that device is missing or busy, use the default mic
        (before, a missing device 1 meant voice input never worked)."""
        settings = dict(samplerate=48000, blocksize=12000, dtype="int16",
                        channels=1, callback=audio_callback)
        try:
            return sd.RawInputStream(device=MIC_DEVICE, **settings)
        except Exception as e:
            self.log_message(f"System: Microphone {MIC_DEVICE} unavailable ({e}); using the default microphone.")
            return sd.RawInputStream(device=None, **settings)

    def background_listener(self):
        self._wait_for_window()
        self.update_status("🎙️ CORE: STANDBY MODE", "#00E5FF", "#1E293B")
        # The offline speech model often mishears unusual words, so also
        # accept the spellings it is likely to produce for "Kooka".
        wake_word_variations = [
            "kooka",
            "cooker",
            "kuka",
            "cuca",
            "cook a"
        ]

        try:
            with self._open_microphone():
                while self.is_running:
                    data = audio_queue.get()

                    # Ignore the microphone while a video plays, so the video's
                    # own sound isn't treated as speech. Use the buttons instead.
                    if self._video_state() == "playing":
                        continue

                    # Ignore the microphone while Kooka is talking, so it doesn't
                    # hear its own voice (e.g. after a typed message).
                    if self.is_speaking:
                        continue

                    # Follow-up window: after a reply you can keep talking without
                    # the wake word for a few seconds, then it goes back to standby.
                    # (Before, this timeout was never applied.)
                    if (self.session_active and not self.enrollment_active
                            and time.time() - self.last_interaction_time > self.session_timeout):
                        self.session_active = False
                        self.update_status("🎙️ CORE: STANDBY MODE", "#00E5FF", "#1E293B")

                    if self.recognizer.AcceptWaveform(data):
                        res_type = "final"
                        result = json.loads(self.recognizer.Result())
                    else:
                        res_type = "partial"
                        result = json.loads(self.recognizer.PartialResult())

                    text = result.get(
                        "text" if res_type == "final" else "partial", ""
                    ).lower().strip()

                    # Still talking? Keep the follow-up window open until you finish.
                    if text and self.session_active:
                        self.last_interaction_time = time.time()

                    # Enrollment is handled before normal commands. Only
                    # completed utterances with a speaker vector count.
                    if self.enrollment_active:
                        if res_type == "final":
                            speaker_data = result.get("spk")
                            if speaker_data is not None:
                                vector = np.asarray(speaker_data, dtype=np.float32)
                                if np.linalg.norm(vector) > 0:
                                    with self.enrollment_lock:
                                        if self.enrollment_waiting:
                                            self.enrollment_samples.append(vector)
                                            self.enrollment_waiting = False
                                            self.enrollment_sample_event.set()
                            self.recognizer.Reset()
                        continue

                    # Do not trigger commands from partial transcripts.
                    if res_type != "final" or not text:
                        continue

                    # Fail closed: a profile and a speaker embedding are both
                    # required before any spoken command can be processed.
                    if self.owner_voice_vector is None:
                        self.update_status("🔒 ENROLL VOICE PROFILE FIRST", "#F59E0B", "#F59E0B")
                        continue

                    speaker_data = result.get("spk")
                    if speaker_data is None or not hasattr(self, "spk_model"):
                        self.log_message("System: No speaker embedding available; voice input ignored.")
                        self.recognizer.Reset()
                        continue

                    current_spk_vector = np.asarray(speaker_data, dtype=np.float32)
                    owner_vector = np.asarray(self.owner_voice_vector, dtype=np.float32)

                    if current_spk_vector.shape != owner_vector.shape:
                        self.log_message("System: Speaker profile dimensions do not match; voice input ignored.")
                        self.recognizer.Reset()
                        continue

                    norm_a = np.linalg.norm(owner_vector)
                    norm_b = np.linalg.norm(current_spk_vector)
                    if norm_a == 0 or norm_b == 0:
                        self.recognizer.Reset()
                        continue

                    similarity = float(
                        np.dot(owner_vector, current_spk_vector) / (norm_a * norm_b)
                    )

                    # This is a starting threshold, not a guaranteed identity
                    # check. Calibrate it using genuine and non-owner samples.
                    BIOMETRIC_MIN_SIMILARITY = 0.20
                    if similarity < BIOMETRIC_MIN_SIMILARITY:
                        self.log_message(
                            f"System: Voice did not match the enrolled profile "
                            f"(score {similarity:.3f}); command ignored."
                        )
                        self.recognizer.Reset()
                        continue

                    print(f"VERIFIED SPEECH: {text} (similarity={similarity:.3f})")

                    detected_wake_word = None
                    for variation in wake_word_variations:
                        if variation in text:
                            detected_wake_word = variation
                            break

                    if self.session_active:
                        if detected_wake_word:
                            text = text.replace(detected_wake_word, "", 1)
                        text = text.strip()

                        if not text:
                            continue

                        if text in ["bye", "goodbye", "go to sleep", "standby", "bi"]:
                            self.session_active = False
                            self.speak("Understood, sir. Shifting back to standby mode.")
                            self.session_active = False
                            self.update_status("🎙️ CORE: STANDBY MODE", "#00E5FF", "#1E293B")
                            self.log_message("System: User terminated link. Standing by.")
                            self.recognizer.Reset()
                            continue

                        self.log_message(f"You (Follow-up): {text}")
                        ai_response = self.ask_groq_brain(text)
                        self.speak(ai_response)

                    elif detected_wake_word:
                        command = text.split(detected_wake_word, 1)[-1].strip()
                        self.recognizer.Reset()

                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break

                        self.log_message("System: Enrolled voice matched.")
                        if not command:
                            self.speak("Core systems online. What are your directives, sir?")
                        else:
                            self.log_message(f"You: {command}")
                            ai_response = self.ask_groq_brain(command)
                            self.speak(ai_response)

        except Exception as e:
            self.log_message(f"Microphone listener stopped: {e}")
            self.update_status("⚠️ MICROPHONE LISTENER ERROR", "#EF4444", "#EF4444")

if __name__ == "__main__":
    root = ctk.CTk()  
    app = AgentBlueApp(root)
    root.mainloop()