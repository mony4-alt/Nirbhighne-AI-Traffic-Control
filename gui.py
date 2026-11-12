"""
gui.py — Main Application Window
Minimalistic dark UI built with CustomTkinter.
Three-column layout: sidebar | video canvas | stats panel.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk

try:
    import tkinterdnd2
    _DND_OK = True
except ImportError:
    _DND_OK = False

import config
import theme
import utils
from utils import logger
from detector import VehicleDetector
from evaluator import (
    run_evaluation,
    run_standard_evaluation,
    generate_custom_evaluation_charts,
    EvaluationResult,
)
from youtube import YouTubeDownloader, DownloadCancelledException
from emergency_overlay import EmergencyBanner, SignalTimerRow


# ─────────────────────────────────────────────────────────────────────────────
# Shared micro-widgets
# ─────────────────────────────────────────────────────────────────────────────

def _sep(parent: ctk.CTkBaseClass, padx: int = 0, pady: tuple = (6, 6)):
    """Draw a 1-px horizontal separator."""
    ctk.CTkFrame(parent, height=1, fg_color=theme.BORDER).pack(
        fill="x", padx=padx, pady=pady
    )


def _section(parent: ctk.CTkBaseClass, text: str):
    """ALL-CAPS muted section header label."""
    ctk.CTkLabel(
        parent, text=text,
        font=theme.caption_bold(),
        text_color=theme.TEXT_3,
        anchor="w"
    ).pack(fill="x", padx=16, pady=(10, 3))


# ─────────────────────────────────────────────────────────────────────────────
# AppGUI
# ─────────────────────────────────────────────────────────────────────────────

class AppGUI(ctk.CTk):
    """Main application window — minimalistic three-column layout."""

    def __init__(self):
        super().__init__()

        # ── DnD initialisation ────────────────────────────────────────────────
        self.dnd_ok = False
        if _DND_OK:
            try:
                tkinterdnd2.TkinterDnD._require(self)
                self.dnd_ok = True
            except Exception as exc:
                logger.warning(f"tkinterdnd2 unavailable: {exc}")

        # ── Window basics ─────────────────────────────────────────────────────
        self.title(config.APP_TITLE)
        self.geometry(f"{config.DEFAULT_WINDOW_WIDTH}x{config.DEFAULT_WINDOW_HEIGHT}")
        self.minsize(960, 620)
        self.configure(fg_color=theme.BG_ROOT)

        if os.path.exists(config.APP_ICON_ICO):
            try:
                self.iconbitmap(config.APP_ICON_ICO)
            except Exception:
                pass

        # ── Runtime state ─────────────────────────────────────────────────────

        self.model_loaded: bool = False
        self.lane_video_paths: list[str | None] = [None]
        self._lane_slot_btns: list[ctk.CTkButton] = []
        self._lane_slot_lbls: list[ctk.CTkLabel] = []
        # Per-lane YouTube buttons are kept in a separate list so downloads
        # can be locked and restored independently from Browse controls.
        self._lane_yt_btns:   list[ctk.CTkButton] = []
        self.youtube_downloader: YouTubeDownloader | None = None
        # Per-lane YouTube state (lane_index → downloader / thread)
        self._lane_yt_downloaders: dict[int, YouTubeDownloader] = {}
        self._lane_yt_threads:     dict[int, threading.Thread]  = {}
        self.yt_thread: threading.Thread | None = None
        self._yt_downloading: bool = False
        self._fullscreen_win = None
        self._fullscreen_lbl = None
        self._eval_thread: threading.Thread | None = None  # evaluation worker
        self._last_frame_sequence: int = -1
        # Emergency banner state
        self._emergency_banner_visible: bool = False
        # Signal timer row widgets: lane → SignalTimerRow
        self._signal_timer_rows: dict[int, SignalTimerRow] = {}
        # Guard to prevent re-entrant device-switch calls when StringVar is
        # updated programmatically inside _check_model / _on_device_change.
        self._device_change_in_progress: bool = False
        # Camera detection state
        self._camera_active: bool = False
        self._youtube_target_lane: int = 0

        # ── Core engine ───────────────────────────────────────────────────────
        self.detector = VehicleDetector()

        # ── History ───────────────────────────────────────────────────────────
        self._history_file = os.path.join(config.LOGS_DIR, "history.txt")
        self._history: list[str] = []

        # ── Build ─────────────────────────────────────────────────────────────
        self._build()
        # Start in the one-lane layout; lane-specific source cards are only
        # useful once the user selects a multi-lane mode.
        self._on_lane_mode_change(self._lane_mode.get())
        self._load_history()
        self._check_model()
        self._poll()

        # Keyboard Shortcuts
        self.bind("<space>", lambda e: self.toggle_pause())
        self.bind("<Control-o>", lambda e: self.browse_file())
        self.bind("<Control-O>", lambda e: self.browse_file())
        self.bind("<Escape>", lambda e: self.close_fullscreen())

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _check_model(self):
        """Load the YOLO model on a background thread to avoid freezing the UI."""
        if not os.path.exists(config.MODEL_PATH):
            self.after(700, lambda: self._dialog(
                "Model Missing",
                f"No model file found.\n\nExpected location:\n{config.MODEL_PATH}\n\n"
                "Place  weights/best.pt  in the application folder\nto enable detection."
            ))
            return

        def _load_worker():
            initial_device = self._selected_device.get()
            try:
                self.detector.load_model(device=initial_device)
                _, dev_name = utils.get_device_info(self.detector.device)
                def _on_done():
                    self.model_loaded = True
                    self._set_device_var_safe(self.detector.device)
                    self._dev_val.configure(text=dev_name, text_color=theme.SUCCESS)
                    self._update_device_ui()
                self.after(0, _on_done)
            except Exception as exc:
                err_msg = str(exc)
                def _on_err():
                    self._dev_val.configure(text="Load failed", text_color=theme.DANGER)
                    self._dialog("Model Error", f"Failed to load model:\n{err_msg}")
                self.after(0, _on_err)

        threading.Thread(target=_load_worker, name="ModelLoader", daemon=True).start()

    def _set_device_var_safe(self, value: str):
        """Set the device StringVar without triggering _on_device_change."""
        self._device_change_in_progress = True
        try:
            self._selected_device.set(value)
        finally:
            self._device_change_in_progress = False

    def _on_device_change(self):
        """Called when the user switches CPU/GPU radio button.

        Reloads the model on the newly selected device.
        The radio button is disabled during reload to prevent double-clicks.
        """
        # Ignore programmatic StringVar updates triggered inside _check_model
        # or _on_device_change itself (e.g. set via _set_device_var_safe).
        if self._device_change_in_progress:
            return

        if not self.model_loaded or self.detector.is_busy:
            return

        new_device = self._selected_device.get()
        if new_device == self.detector.device:
            return  # Nothing to do

        logger.info(f"Device switch requested: {self.detector.device} → {new_device}")
        self._dev_val.configure(text="Switching…", text_color=theme.WARNING)
        self._rb_cpu.configure(state="disabled")
        self._rb_gpu.configure(state="disabled")
        self.model_loaded = False

        def _reload_worker():
            try:
                self.detector.load_model(device=new_device)
                _, dev_name = utils.get_device_info(self.detector.device)
                def _done():
                    self.model_loaded = True
                    self._set_device_var_safe(self.detector.device)
                    self._dev_val.configure(text=dev_name, text_color=theme.SUCCESS)
                    self._update_device_ui()
                self.after(0, _done)
            except Exception as exc:
                err_msg = str(exc)
                def _err():
                    self._dev_val.configure(text="Switch failed", text_color=theme.DANGER)
                    self._update_device_ui()
                    self._dialog("Device Error", f"Could not switch device:\n{err_msg}")
                self.after(0, _err)

        threading.Thread(target=_reload_worker, name="DeviceReload", daemon=True).start()

    def _update_device_ui(self):
        """Re-enable device choices after model load."""
        busy = self.detector.is_busy
        cpu_state = "disabled" if busy else "normal"
        self._rb_cpu.configure(state=cpu_state)
        accelerator = self.detector.preferred_accelerator()
        if accelerator:
            self._rb_gpu.configure(
                text=(
                    "GPU (Intel OpenVINO)" if accelerator == "intel:gpu"
                    else "GPU (Windows DirectML)" if accelerator == "directml"
                    else "GPU (NVIDIA CUDA)"
                ),
                text_color=theme.TEXT_1,
                value=accelerator,
                state=cpu_state,
            )
        else:
            self._rb_gpu.configure(
                text="GPU  —  CUDA N/A",
                text_color=theme.TEXT_3,
                state="disabled",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Layout assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0)   # sidebar — fixed
        self.grid_columnconfigure(1, weight=1)   # canvas  — stretches
        self.grid_columnconfigure(2, weight=0)   # stats   — fixed
        self._build_sidebar()
        self._build_main()
        self._build_stats()
        self._build_footer()

    # ─── Sidebar (column 0) ───────────────────────────────────────────────────

    def _build_sidebar(self):
        sb = ctk.CTkFrame(
            self, width=252,
            fg_color=theme.BG_SURFACE,
            corner_radius=0
        )
        sb.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sb.pack_propagate(False)

        # Brand header with logo icon
        brand_frame = ctk.CTkFrame(sb, fg_color="transparent")
        brand_frame.pack(fill="x", padx=14, pady=(16, 4))
        brand_frame.grid_columnconfigure(1, weight=1)

        if os.path.exists(config.APP_ICON_PNG):
            try:
                pil_icon = Image.open(config.APP_ICON_PNG)
                self._sidebar_logo_img = ctk.CTkImage(light_image=pil_icon, dark_image=pil_icon, size=(36, 36))
                logo_lbl = ctk.CTkLabel(brand_frame, image=self._sidebar_logo_img, text="")
                logo_lbl.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="center")
            except Exception:
                pass

        ctk.CTkLabel(
            brand_frame, text="NIRBHIGHNE",
            font=ctk.CTkFont("Segoe UI", 13, "bold"), text_color=theme.ACCENT, anchor="w"
        ).grid(row=0, column=1, sticky="sw", pady=(0, 1))

        ctk.CTkLabel(
            brand_frame, text="AI-Driven Traffic Signal Control",
            font=ctk.CTkFont("Segoe UI", 8, "bold"), text_color=theme.TEXT_3, anchor="w"
        ).grid(row=1, column=1, sticky="nw", pady=(1, 0))

        _sep(sb, padx=16, pady=(10, 4))



        # ── Video source ──────────────────────────────────────────────────────
        _section(sb, "VIDEO SOURCE")

        # Drop zone
        self._dz = ctk.CTkFrame(
            sb, height=68,
            fg_color=theme.BG_ELEVATED,
            border_color=theme.BORDER,
            border_width=1,
            corner_radius=theme.R_MD
        )
        self._dz.pack(fill="x", padx=16, pady=(0, 8))
        self._dz.pack_propagate(False)

        self._dz_lbl = ctk.CTkLabel(
            self._dz,
            text="Drop video here" if self.dnd_ok else "Click Browse to open",
            font=theme.caption(), text_color=theme.TEXT_3
        )
        self._dz_lbl.pack(expand=True)

        if self.dnd_ok:
            try:
                from tkinterdnd2 import DND_FILES
                for w in (self._dz, self._dz_lbl):
                    w.drop_target_register(DND_FILES)
                    w.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

        for w in (self._dz, self._dz_lbl):
            w.bind("<Double-Button-1>", lambda e: self.browse_file())

        # Browse button
        self._browse_btn = ctk.CTkButton(
            sb, text="Browse Video(s)",
            font=theme.label_bold(),
            fg_color="transparent",
            text_color=theme.TEXT_1,
            border_color=theme.BORDER, border_width=1,
            hover_color=theme.BG_ELEVATED,
            corner_radius=theme.R_MD, height=32,
            command=self.browse_file
        )
        self._browse_btn.pack(fill="x", padx=16, pady=(0, 4))

        # Lane-specific source cards are shown only for 2/4-lane modes.
        self._lane_slots_frame = ctk.CTkFrame(sb, fg_color="transparent")
        self._lane_slots_frame.pack(fill="x", padx=16, pady=(0, 4))
        self._rebuild_lane_slots(1)

        self._lane_help_lbl = ctk.CTkLabel(
            sb,
            text="Choose a video or YouTube source for each lane.",
            font=theme.caption(), text_color=theme.TEXT_3, anchor="w",
            wraplength=200, justify="left",
        )
        self._lane_help_lbl.pack(fill="x", padx=16, pady=(0, 4))

        # ── YouTube ───────────────────────────────────────────────────────────
        # YouTube input is part of the unified video-source area above.

        self._yt_entry = ctk.CTkEntry(
            sb,
            placeholder_text="Paste YouTube URL…",
            fg_color=theme.BG_ELEVATED,
            border_color=theme.BORDER, border_width=1,
            text_color=theme.TEXT_1,
            placeholder_text_color=theme.TEXT_3,
            corner_radius=theme.R_MD,
            font=theme.label(), height=32
        )
        self._yt_entry.pack(fill="x", padx=16, pady=(0, 6))

        self._yt_btn = ctk.CTkButton(
            sb, text="YT Detect",
            font=theme.label_bold(),
            fg_color="transparent",
            text_color=theme.ACCENT,
            border_color=theme.ACCENT, border_width=1,
            hover_color=theme.BG_ELEVATED,
            corner_radius=theme.R_MD, height=32,
            command=self.start_youtube_flow
        )
        self._yt_btn.pack(fill="x", padx=16, pady=(0, 4))

        # Reorder the YouTube field into the unified source area and
        # place the two primary source actions directly beneath it.
        self._browse_btn.destroy()
        self._yt_btn.destroy()
        self._yt_entry.pack_forget()
        self._yt_entry.pack(
            before=self._lane_slots_frame,
            fill="x", padx=16, pady=(0, 6),
        )
        source_actions = ctk.CTkFrame(sb, fg_color="transparent")
        self._source_actions = source_actions
        source_actions.pack(before=self._lane_slots_frame, fill="x", padx=16, pady=(0, 4))
        source_actions.grid_columnconfigure(0, weight=1)
        source_actions.grid_columnconfigure(1, weight=1)
        self._browse_btn = ctk.CTkButton(
            source_actions, text="Browse Videos", font=theme.label_bold(),
            fg_color="transparent", text_color=theme.TEXT_1,
            border_color=theme.BORDER, border_width=1,
            hover_color=theme.BG_ELEVATED, corner_radius=theme.R_MD, height=32,
            command=self.browse_file,
        )
        self._browse_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self._yt_btn = ctk.CTkButton(
            source_actions, text="YT Detect", font=theme.label_bold(),
            fg_color="transparent", text_color=theme.ACCENT,
            border_color=theme.ACCENT, border_width=1,
            hover_color=theme.BG_ELEVATED, corner_radius=theme.R_MD, height=32,
            command=self.start_youtube_flow,
        )
        self._yt_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        # Stable pack anchor for switching between source layouts and the
        # camera section below.
        self._source_camera_sep = ctk.CTkFrame(sb, height=1, fg_color=theme.BORDER)
        self._source_camera_sep.pack(fill="x", padx=16, pady=(6, 6))

        # ── Camera ────────────────────────────────────────────────────────────────────
        _section(sb, "CAMERA")

        cam_row = ctk.CTkFrame(sb, fg_color="transparent")
        cam_row.pack(fill="x", padx=16, pady=(0, 4))
        cam_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            cam_row, text="Index:",
            font=theme.caption(), text_color=theme.TEXT_2, anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=(0, 6))

        self._cam_index_var = tk.StringVar(value="0")
        self._cam_index_entry = ctk.CTkEntry(
            cam_row,
            textvariable=self._cam_index_var,
            placeholder_text="0",
            width=54,
            fg_color=theme.BG_ELEVATED,
            border_color=theme.BORDER, border_width=1,
            text_color=theme.TEXT_1,
            corner_radius=theme.R_MD,
            font=theme.label(), height=30
        )
        self._cam_index_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))

        self._cam_scan_btn = ctk.CTkButton(
            cam_row, text="Scan",
            font=theme.caption(),
            fg_color="transparent",
            text_color=theme.TEXT_2,
            border_color=theme.BORDER, border_width=1,
            hover_color=theme.BG_ELEVATED,
            corner_radius=theme.R_MD, height=30, width=48,
            command=self._scan_cameras,
        )
        self._cam_scan_btn.grid(row=0, column=2, sticky="e")

        self._cam_status_lbl = ctk.CTkLabel(
            sb, text="No camera selected",
            font=theme.caption(), text_color=theme.TEXT_3, anchor="w"
        )
        self._cam_status_lbl.pack(fill="x", padx=16, pady=(0, 4))

        self._cam_start_btn = ctk.CTkButton(
            sb, text="📷  Open Camera Detect",
            font=theme.label_bold(),
            fg_color=theme.ACCENT,
            text_color=theme.BG_ROOT,
            hover_color=theme.ACCENT_DIM,
            corner_radius=theme.R_MD, height=32,
            command=self.start_camera_detection
        )
        self._cam_start_btn.pack(fill="x", padx=16, pady=(0, 4))

        self._cctv_btn = ctk.CTkButton(
            sb, text="CCTV Camera",
            font=theme.caption(),
            fg_color="transparent", text_color=theme.ACCENT,
            border_color=theme.ACCENT, border_width=1,
            hover_color=theme.BG_ELEVATED,
            corner_radius=theme.R_MD, height=28,
            command=lambda: None,
        )
        self._cctv_btn.pack(fill="x", padx=16, pady=(0, 4))

        # Put the real camera action and the presentation-only CCTV option in
        # one compact row. Recreate them in this row because Tk widgets cannot
        # be re-parented after construction.
        self._cam_start_btn.pack_forget()
        self._cam_start_btn.destroy()
        self._cctv_btn.pack_forget()
        self._cctv_btn.destroy()
        camera_action_row = ctk.CTkFrame(sb, fg_color="transparent")
        camera_action_row.pack(fill="x", padx=16, pady=(0, 4))
        camera_action_row.grid_columnconfigure(0, weight=1)
        camera_action_row.grid_columnconfigure(1, weight=1)
        self._cam_start_btn = ctk.CTkButton(
            camera_action_row, text="Camera",
            font=theme.label_bold(), fg_color=theme.ACCENT,
            text_color=theme.BG_ROOT, hover_color=theme.ACCENT_DIM,
            corner_radius=theme.R_MD, height=32,
            command=self.start_camera_detection,
        )
        self._cam_start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self._cctv_btn = ctk.CTkButton(
            camera_action_row, text="CCTV Camera",
            font=theme.caption(), fg_color="transparent",
            text_color=theme.ACCENT, border_color=theme.ACCENT,
            border_width=1, hover_color=theme.BG_ELEVATED,
            corner_radius=theme.R_MD, height=32,
            command=lambda: None,
        )
        self._cctv_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        _sep(sb, padx=16)

        # ── History ───────────────────────────────────────────────────────────
        _section(sb, "HISTORY")

        self._hist_lb = tk.Listbox(
            sb,
            bg=theme.BG_ELEVATED,
            fg=theme.TEXT_2,
            selectbackground=theme.ACCENT,
            selectforeground=theme.TEXT_1,
            borderwidth=0, highlightthickness=0,
            relief="flat",
            font=("Segoe UI", 9),
            activestyle="none"
        )
        self._hist_lb.pack(fill="both", expand=True, padx=16, pady=(0, 6))
        self._hist_lb.bind("<Double-Button-1>", self._hist_click)

        self._clear_lbl = ctk.CTkLabel(
            sb, text="Clear history",
            font=theme.caption(),
            text_color=theme.DANGER,
            cursor="hand2", anchor="w"
        )
        self._clear_lbl.pack(fill="x", padx=16, pady=(0, 16))
        self._clear_lbl.bind("<Button-1>", lambda e: self.clear_history())

    # ─── Main / canvas panel (column 1) ──────────────────────────────────────

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=theme.BG_ROOT, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(2, weight=1)  # row 2 = canvas (row 1 = emergency banner)
        main.grid_columnconfigure(0, weight=1)

        # Source header row
        hdr = ctk.CTkFrame(main, fg_color="transparent", height=36)
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 0))
        hdr.grid_columnconfigure(0, weight=1)
        hdr.grid_propagate(False)

        self._source_lbl = ctk.CTkLabel(
            hdr, text="No video loaded",
            font=theme.label_bold(), text_color=theme.TEXT_2, anchor="w"
        )
        self._source_lbl.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(
            hdr, text="Double-click for fullscreen",
            font=theme.caption(), text_color=theme.TEXT_3, anchor="e"
        ).grid(row=0, column=1, sticky="e")

        # ── Emergency banner (initially hidden) ───────────────────────────
        self._emergency_banner = EmergencyBanner(main)
        self._emergency_banner.grid(row=1, column=0, sticky="ew", padx=0, pady=0)
        self._emergency_banner.grid_remove()   # hidden until ambulance confirmed

        # Video canvas — no border, pure black
        self._canvas = tk.Canvas(
            main,
            bg=theme.BG_ROOT, bd=0,
            highlightthickness=0, relief="flat"
        )
        self._canvas.grid(row=2, column=0, sticky="nsew", padx=20, pady=8)
        self._canvas.bind("<Double-Button-1>", self.open_fullscreen)
        self._canvas.bind("<Configure>", lambda e: self._redraw())

        # Control bar
        ctrl = ctk.CTkFrame(main, fg_color="transparent", height=42)
        ctrl.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 6))
        ctrl.grid_propagate(False)

        def _btn(text, color, hover, cmd, w=88):
            return ctk.CTkButton(
                ctrl, text=text,
                font=theme.label_bold(),
                fg_color=color, hover_color=hover,
                corner_radius=theme.R_MD, height=34, width=w,
                command=cmd
            )

        self._start_btn  = _btn("▶  Start",   theme.SUCCESS,  "#16a34a", self.start_detection)
        self._pause_btn  = _btn("⏸  Pause",   theme.WARNING,  "#d97706", self.pause_detection)
        self._resume_btn = _btn("▶▶  Resume", theme.ACCENT,   theme.ACCENT_DIM, self.resume_detection)
        self._stop_btn   = _btn("⏹  Stop",    theme.DANGER,   "#dc2626", self.stop_detection)

        self._start_btn.pack(side="left", padx=(0, 5))
        self._pause_btn.pack(side="left", padx=(0, 4))
        self._resume_btn.pack(side="left", padx=(0, 4))
        self._stop_btn.pack(side="left", padx=(0, 14))

        self._pause_btn.configure(state="disabled")
        self._resume_btn.configure(state="disabled")
        self._stop_btn.configure(state="disabled")

        self._save_chk = ctk.CTkCheckBox(
            ctrl, text="Save output",
            font=theme.label(), text_color=theme.TEXT_2,
            fg_color=theme.ACCENT, hover_color=theme.ACCENT_DIM,
            border_color=theme.BORDER,
            corner_radius=3, checkbox_width=15, checkbox_height=15
        )
        self._save_chk.pack(side="left")
        self._save_chk.select()

        self._gray_chk = ctk.CTkCheckBox(
            ctrl, text="Grayscale",
            font=theme.label(), text_color=theme.TEXT_2,
            fg_color=theme.ACCENT, hover_color=theme.ACCENT_DIM,
            border_color=theme.BORDER,
            corner_radius=3, checkbox_width=15, checkbox_height=15,
            command=self._on_grayscale_toggle
        )
        self._gray_chk.pack(side="left", padx=(12, 0))

        ctk.CTkButton(
            ctrl, text="Open Output",
            font=theme.caption(),
            fg_color="transparent", text_color=theme.TEXT_2,
            hover_color=theme.BG_ELEVATED,
            border_width=0,
            corner_radius=theme.R_MD, height=28,
            command=self.open_output_folder
        ).pack(side="right")

        self._eval_btn = ctk.CTkButton(
            ctrl, text="📊  Evaluate Model",
            font=theme.label_bold(),
            fg_color="transparent",
            text_color=theme.ACCENT,
            border_color=theme.ACCENT, border_width=1,
            hover_color=theme.BG_ELEVATED,
            corner_radius=theme.R_MD, height=34, width=130,
            command=self.open_eval_dialog
        )
        self._eval_btn.pack(side="right", padx=(0, 8))

        # Progress row (ultra-thin bar)
        prog = ctk.CTkFrame(main, fg_color="transparent", height=26)
        prog.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 12))
        prog.grid_columnconfigure(1, weight=1)
        prog.grid_propagate(False)

        self._state_lbl = ctk.CTkLabel(
            prog, text="Idle",
            font=theme.caption(), text_color=theme.TEXT_3, anchor="w", width=60
        )
        self._state_lbl.grid(row=0, column=0, sticky="w")

        self._pct_lbl = ctk.CTkLabel(
            prog, text="0%",
            font=theme.mono(), text_color=theme.TEXT_3, anchor="e", width=42
        )
        self._pct_lbl.grid(row=0, column=2, sticky="e")

        self._progress = ctk.CTkProgressBar(
            prog,
            fg_color=theme.BG_ELEVATED,
            progress_color=theme.ACCENT,
            corner_radius=1, height=3
        )
        self._progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        self._progress.set(0)

    # ─── Stats panel (column 2) ───────────────────────────────────────────────

    def _build_stats(self):
        sp = ctk.CTkFrame(
            self, width=262,
            fg_color=theme.BG_SURFACE,
            corner_radius=0
        )
        sp.grid(row=0, column=2, sticky="nsew")
        sp.pack_propagate(False)

        inner = ctk.CTkScrollableFrame(
            sp, fg_color="transparent",
            scrollbar_button_color=theme.BG_ELEVATED,
            scrollbar_button_hover_color=theme.BORDER
        )
        inner.pack(fill="both", expand=True, pady=(10, 0))

        # ── Device selection ──────────────────────────────────────────────────
        _section(inner, "DEVICE")

        accelerator = self.detector.preferred_accelerator()
        self._selected_device = tk.StringVar(value=accelerator or "cpu")
        cuda_ok = bool(accelerator)

        # Radio: CPU
        self._rb_cpu = ctk.CTkRadioButton(
            inner, text="CPU",
            variable=self._selected_device, value="cpu",
            font=theme.label(), text_color=theme.TEXT_1,
            fg_color=theme.ACCENT, hover_color=theme.ACCENT_DIM,
            border_color=theme.BORDER,
            command=self._on_device_change
        )
        self._rb_cpu.pack(fill="x", padx=16, pady=(2, 2))

        # Radio: GPU (CUDA)
        gpu_label = "GPU (CUDA)" if cuda_ok else "GPU  —  CUDA N/A"
        gpu_color = theme.TEXT_1 if cuda_ok else theme.TEXT_3
        gpu_state = "normal" if cuda_ok else "disabled"
        self._rb_gpu = ctk.CTkRadioButton(
            inner, text=gpu_label,
            variable=self._selected_device, value=accelerator or "cpu",
            font=theme.label(), text_color=gpu_color,
            fg_color=theme.ACCENT, hover_color=theme.ACCENT_DIM,
            border_color=theme.BORDER,
            state=gpu_state,
            command=self._on_device_change
        )
        if accelerator == "intel:gpu":
            self._rb_gpu.configure(text="GPU (Intel OpenVINO)")
        elif accelerator == "directml":
            self._rb_gpu.configure(text="GPU (Windows DirectML)")
        self._rb_gpu.pack(fill="x", padx=16, pady=(2, 6))

        self._dev_val = ctk.CTkLabel(
            inner, text="Loading model…",
            font=theme.label_bold(), text_color=theme.WARNING, anchor="w"
        )
        self._dev_val.pack(fill="x", padx=16, pady=(0, 6))

        _sep(inner, padx=16)

        # ── Detection Sensitivity (Confidence Slider) ──────────────────────────
        _section(inner, "SENSITIVITY (CONFIDENCE)")
        self._conf_val_lbl = ctk.CTkLabel(
            inner,
            text=f"Min Confidence: {int(config.DEFAULT_CONFIDENCE * 100)}%",
            font=theme.caption(), text_color=theme.TEXT_2, anchor="w"
        )
        self._conf_val_lbl.pack(fill="x", padx=16, pady=(0, 2))

        self._conf_slider = ctk.CTkSlider(
            inner,
            from_=15, to=85, number_of_steps=70,
            fg_color=theme.BG_ELEVATED,
            progress_color=theme.ACCENT,
            button_color=theme.ACCENT,
            button_hover_color=theme.ACCENT_DIM,
            command=self._on_conf_slider_change
        )
        self._conf_slider.set(int(config.DEFAULT_CONFIDENCE * 100))
        self._conf_slider.pack(fill="x", padx=16, pady=(0, 6))

        _sep(inner, padx=16)

        # Lane-zone selection and traffic recommendations
        _section(inner, "LANE & TRAFFIC")
        self._lane_mode = tk.StringVar(value="1 lane")
        self._lane_selector = ctk.CTkOptionMenu(
            inner,
            values=["1 lane", "2 lanes", "4 lanes"],
            variable=self._lane_mode,
            command=self._on_lane_mode_change,
            font=theme.label(),
            fg_color=theme.BG_ELEVATED,
            button_color=theme.ACCENT,
            button_hover_color=theme.ACCENT_DIM,
            text_color=theme.TEXT_1,
            dropdown_fg_color=theme.BG_ELEVATED,
            dropdown_hover_color=theme.BORDER,
            height=32
        )
        self._lane_selector.pack(fill="x", padx=16, pady=(0, 5))

        self._lane_counts_val = ctk.CTkLabel(
            inner, text="Lane 1: 0 vehicles",
            font=theme.mono(), text_color=theme.TEXT_2, anchor="w", wraplength=225
        )
        self._lane_counts_val.pack(fill="x", padx=16, pady=(0, 3))
        self._signal_val = ctk.CTkLabel(
            inner, text="Signal advice will appear here",
            font=theme.label_bold(), text_color=theme.TEXT_2, anchor="w", wraplength=225
        )
        self._signal_val.pack(fill="x", padx=16, pady=(0, 3))
        self._emergency_val = ctk.CTkLabel(
            inner, text="No ambulance detected",
            font=theme.label_bold(), text_color=theme.TEXT_3, anchor="w", wraplength=225
        )
        self._emergency_val.pack(fill="x", padx=16, pady=(0, 6))

        _sep(inner, padx=16)

        # ── Per-lane signal timer rows ─────────────────────────────────────────
        _section(inner, "SIGNAL TIMERS")
        self._signal_timer_container = ctk.CTkFrame(inner, fg_color="transparent")
        self._signal_timer_container.pack(fill="x", padx=16, pady=(0, 6))
        self._signal_timer_rows: dict[int, SignalTimerRow] = {}
        self._rebuild_signal_timer_rows(1)

        _sep(inner, padx=16)

        # Performance metrics
        _section(inner, "PERFORMANCE")
        self._perf: dict[str, ctk.CTkLabel] = {}
        for key, lbl_text in [
            ("fps",   "FPS"),
            ("inf",   "Inference"),
            ("frame", "Frame"),
            ("time",  "Elapsed"),
            ("res",   "Resolution"),
        ]:
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=2)
            ctk.CTkLabel(
                row, text=lbl_text,
                font=theme.label(), text_color=theme.TEXT_2, anchor="w"
            ).pack(side="left")
            val = ctk.CTkLabel(
                row, text="—",
                font=theme.mono(), text_color=theme.TEXT_1, anchor="e"
            )
            val.pack(side="right")
            self._perf[key] = val

        _sep(inner, padx=16)

        # Vehicle counts
        _section(inner, "VEHICLES")

        cnt_card = ctk.CTkFrame(
            inner,
            fg_color=theme.BG_ELEVATED,
            corner_radius=theme.R_MD
        )
        cnt_card.pack(fill="x", padx=16, pady=(0, 4))
        cnt_card.grid_columnconfigure((0, 1), weight=1)

        for col, lbl_text, attr in [
            (0, "NOW",   "_cur_lbl"),
            (1, "TOTAL", "_tot_lbl"),
        ]:
            cell = ctk.CTkFrame(cnt_card, fg_color="transparent")
            cell.grid(row=0, column=col, padx=8, pady=10, sticky="nsew")
            ctk.CTkLabel(
                cell, text=lbl_text,
                font=theme.caption(), text_color=theme.TEXT_3
            ).pack()
            num = ctk.CTkLabel(
                cell, text="0",
                font=theme.mono_lg(),
                text_color=theme.ACCENT if col == 0 else theme.SUCCESS
            )
            num.pack()
            setattr(self, attr, num)

        # Traffic Congestion Density Meter
        cong_frame = ctk.CTkFrame(cnt_card, fg_color="transparent")
        cong_frame.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 10), sticky="ew")

        self._congestion_lbl = ctk.CTkLabel(
            cong_frame, text="Traffic: Low Density",
            font=theme.caption(), text_color=theme.SUCCESS, anchor="w"
        )
        self._congestion_lbl.pack(fill="x", pady=(0, 2))

        self._congestion_bar = ctk.CTkProgressBar(
            cong_frame,
            fg_color=theme.BG_SURFACE,
            progress_color=theme.SUCCESS,
            corner_radius=2, height=4
        )
        self._congestion_bar.pack(fill="x")
        self._congestion_bar.set(0.05)

        _sep(inner, padx=16)
        _section(inner, "CLASS BREAKDOWN")

        # Container for dynamic class rows
        self._cls_frame = ctk.CTkFrame(inner, fg_color="transparent")
        self._cls_frame.pack(fill="x", padx=16, pady=(0, 16))
        self._cls_widgets: dict[str, tuple] = {}  # cls -> (row_frame, val_label)

    # ─── Footer (row 1, columns 1-2) ─────────────────────────────────────────

    def _build_footer(self):
        ft = ctk.CTkFrame(
            self, height=24,
            fg_color=theme.BG_SURFACE, corner_radius=0
        )
        ft.grid(row=1, column=1, columnspan=2, sticky="ew")
        ft.grid_propagate(False)

        ctk.CTkLabel(
            ft,
            text="Nirbhighne  ·  Dynamic Traffic Signal Control",
            font=theme.caption(), text_color=theme.TEXT_3
        ).pack(side="left", padx=14)

        self._footer_lbl = ctk.CTkLabel(
            ft, text="Ready",
            font=theme.caption(), text_color=theme.TEXT_3
        )
        self._footer_lbl.pack(side="right", padx=14)

    # ─────────────────────────────────────────────────────────────────────────
    # Modal dialog
    # ─────────────────────────────────────────────────────────────────────────

    def _dialog(self, title: str, message: str):
        dlg = ctk.CTkToplevel(self)
        dlg.title(title)
        dlg.geometry("400x200")
        dlg.resizable(False, False)
        dlg.configure(fg_color=theme.BG_SURFACE)
        dlg.transient(self)
        dlg.grab_set()
        dlg.update_idletasks()
        x = self.winfo_x() + self.winfo_width() // 2 - 200
        y = self.winfo_y() + self.winfo_height() // 2 - 100
        dlg.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            dlg, text=title.upper(),
            font=theme.label_bold(), text_color=theme.WARNING
        ).pack(pady=(20, 8))

        ctk.CTkLabel(
            dlg, text=message,
            font=theme.label(), text_color=theme.TEXT_1,
            wraplength=360, justify="center"
        ).pack(padx=20, pady=(0, 14))

        ctk.CTkButton(
            dlg, text="OK",
            font=theme.label_bold(),
            fg_color=theme.ACCENT, hover_color=theme.ACCENT_DIM,
            corner_radius=theme.R_MD, height=32, width=80,
            command=dlg.destroy
        ).pack()

    # ─────────────────────────────────────────────────────────────────────────
    # Lane video slots
    # ─────────────────────────────────────────────────────────────────────────

    def _lane_count_from_mode(self) -> int:
        return int(self._lane_mode.get().split()[0])

    def _rebuild_lane_slots(self, lane_count: int):
        """Rebuild sidebar lane upload buttons for the selected lane mode."""
        old_paths = list(self.lane_video_paths)
        self.lane_video_paths = []
        for lane in range(lane_count):
            self.lane_video_paths.append(old_paths[lane] if lane < len(old_paths) else None)

        # Each lane also owns a row frame.  Destroy every child so rebuilding
        # (which also happens when detection starts) cannot leave empty rows
        # behind and push the upload controls out of the sidebar.
        for w in self._lane_slots_frame.winfo_children():
            w.destroy()
        self._lane_slot_btns.clear()
        self._lane_slot_lbls.clear()
        self._lane_yt_btns.clear()

        for lane in range(1, lane_count + 1):
            idx = lane - 1
            lbl = ctk.CTkLabel(
                self._lane_slots_frame,
                text=f"LANE {lane}  |  No video selected",
                font=theme.caption(),
                text_color=theme.TEXT_3,
                anchor="w",
            )
            lbl.pack(fill="x", pady=(2, 0))

            # Multi-lane mode gives each feed its own source controls so the
            # mapping between a file/URL and its lane is always explicit.
            row_fr = ctk.CTkFrame(self._lane_slots_frame, fg_color="transparent")
            row_fr.pack(fill="x", pady=(0, 4))
            row_fr.grid_columnconfigure(0, weight=1)
            row_fr.grid_columnconfigure(1, weight=1)

            btn = ctk.CTkButton(
                row_fr,
                text=f"📂 Browse",
                font=theme.caption(),
                fg_color="transparent",
                text_color=theme.TEXT_2,
                border_color=theme.BORDER,
                border_width=1,
                hover_color=theme.BG_ELEVATED,
                corner_radius=theme.R_MD,
                height=26,
                command=lambda i=idx: self.browse_lane_file(i),
            )
            btn.grid(row=0, column=0, sticky="ew", padx=(0, 2))

            yt_btn = ctk.CTkButton(
                row_fr,
                text="▼ YouTube",
                font=theme.caption(),
                fg_color="transparent",
                text_color=theme.ACCENT,
                border_color=theme.ACCENT,
                border_width=1,
                hover_color=theme.BG_ELEVATED,
                corner_radius=theme.R_MD,
                height=26,
                command=lambda i=idx: self._start_lane_yt_download(i),
            )
            yt_btn.configure(text="YT Detect")
            if lane_count == 1:
                # The complete one-lane source layout already has the shared
                # URL field and primary actions above these cards.
                yt_btn.grid_remove()
                btn.grid_configure(columnspan=2, padx=0)
            else:
                yt_btn.grid(row=0, column=1, sticky="ew", padx=(2, 0))

            self._lane_slot_lbls.append(lbl)
            self._lane_slot_btns.append(btn)
            self._lane_yt_btns.append(yt_btn)
            if self.lane_video_paths[idx]:
                self._update_lane_slot_label(idx)

    def _update_lane_slot_label(self, lane_index: int):
        path = self.lane_video_paths[lane_index]
        if lane_index < len(self._lane_slot_lbls):
            text = (
                f"LANE {lane_index + 1}  |  {os.path.basename(path)}"
                if path else f"LANE {lane_index + 1}  |  No video selected"
            )
            self._lane_slot_lbls[lane_index].configure(
                text=text,
                text_color=theme.TEXT_1 if path else theme.TEXT_3,
            )

    def _update_source_header(self):
        loaded = [p for p in self.lane_video_paths if p]
        if not loaded:
            self._source_lbl.configure(text="No videos loaded", text_color=theme.TEXT_2)
            return
        if len(loaded) == 1:
            self._source_lbl.configure(text=os.path.basename(loaded[0]), text_color=theme.TEXT_1)
        else:
            names = ", ".join(os.path.basename(p) for p in loaded)
            self._source_lbl.configure(text=f"{len(loaded)} lane feeds: {names}", text_color=theme.TEXT_1)

    def browse_lane_file(self, lane_index: int):
        path = filedialog.askopenfilename(
            title=f"Select Lane {lane_index + 1} Video",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.set_lane_video(lane_index, path)

    def set_lane_video(self, lane_index: int, path: str):
        self.stop_detection()
        self.lane_video_paths[lane_index] = path
        self._update_lane_slot_label(lane_index)
        self._update_source_header()
        self._footer_lbl.configure(text=f"Lane {lane_index + 1}: {os.path.basename(path)}")
        self._state_lbl.configure(text="Loaded")
        self._reset_display()
        self._canvas.delete("all")
        cw = self._canvas.winfo_width() or 600
        ch = self._canvas.winfo_height() or 400
        self._canvas.create_text(
            cw // 2, ch // 2,
            text="Press  ▶ Start  to begin detection",
            fill=theme.TEXT_3, font=("Segoe UI", 12),
        )
        self._add_history(path)

    def _loaded_lane_paths(self) -> list[str]:
        """Return lane video paths that actually exist on disk.
        Camera sources (int) are excluded from this check.
        """
        return [p for p in self.lane_video_paths if p and os.path.exists(str(p))]

    # ─────────────────────────────────────────────────────────────────────────
    # Drag-and-drop / file handling
    # ─────────────────────────────────────────────────────────────────────────

    def _on_drop(self, event):
        raw = event.data.strip()
        paths = []
        if raw.startswith("{") and "}" in raw:
            # Windows may wrap paths in braces, especially with spaces.
            parts, buf, depth = [], "", 0
            for ch in raw:
                if ch == "{":
                    depth += 1
                    if depth == 1:
                        buf = ""
                        continue
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        parts.append(buf)
                        buf = ""
                        continue
                if depth:
                    buf += ch
            paths = parts
        else:
            paths = raw.split()

        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        videos = [
            os.path.normpath(p.strip('"'))
            for p in paths
            if os.path.splitext(p.lower())[1] in video_exts
        ]
        if not videos:
            self._dialog(
                "Unsupported Format",
                "Only MP4, AVI, MKV, MOV and WEBM files are supported.",
            )
            return

        lane_count = self._lane_count_from_mode()
        for idx, path in enumerate(videos[:lane_count]):
            self.set_lane_video(idx, path)

    def browse_file(self):
        lane_count = self._lane_count_from_mode()
        if lane_count == 1:
            path = filedialog.askopenfilename(
                title="Select a Video File",
                filetypes=[
                    ("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm"),
                    ("All Files", "*.*"),
                ],
            )
            if path:
                self.set_lane_video(0, path)
            return

        paths = filedialog.askopenfilenames(
            title=f"Select up to {lane_count} lane videos",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm"),
                ("All Files", "*.*"),
            ],
        )
        if not paths:
            return
        for idx, path in enumerate(paths[:lane_count]):
            self.set_lane_video(idx, path)

    # ─────────────────────────────────────────────────────────────────────────
    # YouTube download
    # ─────────────────────────────────────────────────────────────────────────

    def _next_empty_lane(self) -> int | None:
        """Choose the next unassigned lane for the shared one-lane action."""
        if len(self.lane_video_paths) == 1:
            return 0
        return next(
            (idx for idx, path in enumerate(self.lane_video_paths) if not path),
            None,
        )

    def start_youtube_flow(self):
        url = self._yt_entry.get().strip()
        if not url:
            self._dialog("Empty URL", "Please paste a YouTube URL first.")
            return
        target_lane = self._next_empty_lane()
        if target_lane is None:
            self._dialog(
                "All Lanes Assigned",
                "All selected lanes already have videos. Replace a lane before downloading another.",
            )
            return
        self._youtube_target_lane = target_lane
        if not utils.check_internet_connection():
            self._dialog("No Network", "Cannot reach the internet.\nCheck your connection and try again.")
            return

        self.stop_detection()
        self._state_lbl.configure(text="Downloading")
        self._footer_lbl.configure(text="Fetching YouTube stream…")
        self._progress.set(0)
        self._pct_lbl.configure(text="0%")
        self._lock_inputs(True)
        self._yt_downloading = True

        self.youtube_downloader = YouTubeDownloader()
        self.yt_thread = threading.Thread(
            target=self._run_yt, args=(url,),
            name="YouTubeDownloader", daemon=True
        )
        self.yt_thread.start()

    def _run_yt(self, url: str):
        try:
            path = self.youtube_downloader.download(
                url, config.OUTPUT_DIR,
                lambda pct, spd, eta, _: self.after(
                    0, lambda: self._yt_progress(pct, spd, eta)
                )
            )
            self.after(0, lambda: self._yt_ok(path))
        except DownloadCancelledException:
            self.after(0, lambda: self._yt_done(cancelled=True))
        except Exception as exc:
            self.after(0, lambda: self._yt_err(str(exc)))

    def _yt_progress(self, pct: float, speed: str, eta: str):
        self._progress.set(pct / 100)
        self._pct_lbl.configure(text=f"{pct:.0f}%")
        self._footer_lbl.configure(text=f"{speed}  ·  ETA {eta}")

    def _yt_ok(self, path: str):
        self._yt_downloading = False
        self._lock_inputs(False)
        self.set_lane_video(self._youtube_target_lane, path)
        self._state_lbl.configure(text="Downloaded")

    def _yt_done(self, cancelled: bool = False):
        self._yt_downloading = False
        self._lock_inputs(False)
        self._progress.set(0)
        self._pct_lbl.configure(text="0%")
        self._state_lbl.configure(text="Cancelled" if cancelled else "Idle")

    def _yt_err(self, msg: str):
        self._yt_downloading = False
        self._lock_inputs(False)
        self._progress.set(0)
        self._pct_lbl.configure(text="0%")
        self._state_lbl.configure(text="Error")
        self._dialog("Download Failed", f"Could not download the video:\n{msg}")

    def _lock_inputs(self, locked: bool):
        state = "disabled" if locked else "normal"
        for w in (self._browse_btn, self._yt_btn, self._start_btn, self._yt_entry, self._lane_selector):
            w.configure(state=state)
        for btn in self._lane_slot_btns + self._lane_yt_btns:
            btn.configure(state=state)
        # Camera controls
        if hasattr(self, "_cam_start_btn"):
            self._cam_start_btn.configure(state=state)
            self._cam_scan_btn.configure(state=state)
            self._cam_index_entry.configure(state=state)
            self._cctv_btn.configure(state=state)
        self._update_device_ui()

    def _set_lane_source_layout(self, lane_count: int):
        """Switch between the shared one-lane source area and lane cards."""
        single_lane = lane_count == 1
        top_source_widgets = (self._dz, self._yt_entry, self._source_actions)

        if single_lane:
            # Keep the drop zone, URL field, and primary actions together.
            for widget in top_source_widgets:
                if widget.winfo_manager() != "pack":
                    widget.pack(
                        before=self._lane_slots_frame,
                        fill="x",
                        padx=16,
                        pady=(0, 8) if widget is self._dz
                        else (0, 6) if widget is self._yt_entry
                        else (0, 4),
                    )
            self._lane_slots_frame.pack_forget()
            self._lane_help_lbl.pack_forget()
        else:
            # Multi-lane mode makes each source-to-lane mapping explicit.
            for widget in top_source_widgets:
                widget.pack_forget()
            if self._lane_slots_frame.winfo_manager() != "pack":
                self._lane_slots_frame.pack(
                    before=self._source_camera_sep,
                    fill="x",
                    padx=16,
                    pady=(0, 4),
                )
            if self._lane_help_lbl.winfo_manager() != "pack":
                self._lane_help_lbl.pack(
                    before=self._source_camera_sep,
                    fill="x",
                    padx=16,
                    pady=(0, 4),
                )

    def _rebuild_signal_timer_rows(self, lane_count: int):
        """Recreate per-lane signal timer rows for the given lane count."""
        for row in self._signal_timer_rows.values():
            row.destroy()
        self._signal_timer_rows.clear()
        for lane in range(1, lane_count + 1):
            row = SignalTimerRow(self._signal_timer_container, lane=lane)
            row.pack(fill="x", pady=2)
            self._signal_timer_rows[lane] = row

    def _on_lane_mode_change(self, value: str):
        """Apply lane mode — each lane uses its own camera video feed."""
        lane_count = int(value.split()[0])
        self._browse_btn.configure(
            text="Browse Video" if lane_count == 1 else f"Browse {lane_count} Videos"
        )
        self._lane_help_lbl.configure(
            text=(
                "Drop a video or paste a YouTube URL."
                if lane_count == 1
                else "Choose a video or YouTube source for each lane."
            )
        )
        self.detector.set_lane_count(lane_count)
        self._rebuild_lane_slots(lane_count)
        self._set_lane_source_layout(lane_count)
        self._rebuild_signal_timer_rows(lane_count)
        self._update_source_header()
        self._lane_counts_val.configure(
            text="  ".join(f"Lane {lane}: 0" for lane in range(1, lane_count + 1))
        )
        self._signal_val.configure(text="Signal advice will appear here", text_color=theme.TEXT_2)
        self._emergency_val.configure(text="No ambulance detected", text_color=theme.TEXT_3)

    def start_detection(self):
        if not self.model_loaded:
            self._dialog("No Model", "weights/best.pt is missing.\nCannot run detection.")
            return

        lane_count = self._lane_count_from_mode()
        paths = self._loaded_lane_paths()
        if len(paths) < lane_count:
            self._dialog(
                "Missing Videos",
                f"{lane_count}-lane mode needs {lane_count} video(s).\n"
                f"Load a separate feed for each lane using the lane buttons.",
            )
            return

        ordered_paths = []
        for lane_path in self.lane_video_paths:
            if lane_path and os.path.exists(lane_path):
                ordered_paths.append(lane_path)
        if len(ordered_paths) != lane_count:
            self._dialog("Missing Videos", "Assign one video to each lane before starting.")
            return

        self.detector.grayscale_mode = bool(self._gray_chk.get())
        self._on_lane_mode_change(self._lane_mode.get())
        self._lock_inputs(True)
        self._update_device_ui()
        self._stop_btn.configure(state="normal")
        self._pause_btn.configure(state="normal")
        self._resume_btn.configure(state="disabled")

        logger.info(f"Starting {lane_count}-lane detection: {ordered_paths}")
        self.detector.start(
            ordered_paths,
            save_output=bool(self._save_chk.get()),
        )

    def pause_detection(self):
        self.detector.pause()
        self._pause_btn.configure(state="disabled")
        self._resume_btn.configure(state="normal")

    def resume_detection(self):
        self.detector.resume()
        self._resume_btn.configure(state="disabled")
        self._pause_btn.configure(state="normal")

    def toggle_pause(self):
        """Spacebar shortcut toggle for pause/resume."""
        if self.detector.current_state == "Processing":
            self.pause_detection()
        elif self.detector.current_state == "Paused":
            self.resume_detection()

    def stop_detection(self):
        if self._yt_downloading and self.youtube_downloader:
            self.youtube_downloader.cancel()
        self.detector.stop()
        self._camera_active = False
        self._lock_inputs(False)
        self._update_device_ui()
        for btn in (self._stop_btn, self._pause_btn, self._resume_btn):
            btn.configure(state="disabled")

    # ─────────────────────────────────────────────────────────────────────────
    # Camera detection
    # ─────────────────────────────────────────────────────────────────────────

    def _scan_cameras(self):
        """Probe camera indices 0-4 and update the status label (fast DSHOW scan)."""
        import cv2 as _cv2
        self._cam_status_lbl.configure(text="Scanning cameras…", text_color=theme.WARNING)
        self.update_idletasks()

        def _worker():
            found = []
            backend = _cv2.CAP_DSHOW if os.name == "nt" else _cv2.CAP_ANY
            for idx in range(5):
                cap = _cv2.VideoCapture(idx, backend)
                if not cap.isOpened() and backend != _cv2.CAP_ANY:
                    cap = _cv2.VideoCapture(idx, _cv2.CAP_ANY)
                if cap.isOpened():
                    found.append(idx)
                cap.release()

            # Schedule GUI update back on main thread
            def _update():
                if found:
                    indices_str = ", ".join(str(i) for i in found)
                    self._cam_status_lbl.configure(
                        text=f"Found: Camera(s) {indices_str}",
                        text_color=theme.SUCCESS,
                    )
                    self._cam_index_var.set(indices_str)
                else:
                    self._cam_status_lbl.configure(
                        text="No cameras found (0-4)",
                        text_color=theme.DANGER,
                    )
            self.after(0, _update)

        threading.Thread(target=_worker, name="CamScanner", daemon=True).start()

    def start_camera_detection(self):
        """Open camera(s) at selected index/indices and start live detection."""
        if not self.model_loaded:
            self._dialog("No Model", "weights/best.pt is missing.\nCannot run detection.")
            return

        raw = self._cam_index_var.get().strip()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        cam_indices = []
        for p in parts:
            try:
                cam_indices.append(int(p))
            except ValueError:
                self._dialog("Invalid Index", f"'{p}' is not a valid camera index.\nEnter integer(s) (e.g. 0 or 0, 1).")
                return

        if not cam_indices:
            self._dialog("Invalid Index", "Please enter at least one camera index (e.g. 0).")
            return

        # Pre-check all requested cameras
        import cv2 as _cv2
        backend = _cv2.CAP_DSHOW if os.name == "nt" else _cv2.CAP_ANY
        failed_cams = []
        for idx in cam_indices:
            test = _cv2.VideoCapture(idx, backend)
            if not test.isOpened() and backend != _cv2.CAP_ANY:
                test = _cv2.VideoCapture(idx, _cv2.CAP_ANY)
            if not test.isOpened():
                failed_cams.append(idx)
            else:
                test.release()

        if failed_cams:
            failed_str = ", ".join(str(i) for i in failed_cams)
            self._dialog(
                "Camera Not Found",
                f"Could not open Camera(s): {failed_str}.\n\n"
                "• Make sure the camera is connected.\n"
                "• Try Scan to find available camera indices.\n"
                "• Check that no other app is using the camera.",
            )
            return

        self.stop_detection()
        self._camera_active = True
        num_cams = len(cam_indices)
        cams_text = ", ".join(str(i) for i in cam_indices)

        self._cam_status_lbl.configure(
            text=f"Camera(s) {cams_text} active", text_color=theme.SUCCESS
        )
        self._source_lbl.configure(
            text=f"📷 Live Camera Feed ({num_cams} feed{'s' if num_cams > 1 else ''}: Cam {cams_text})",
            text_color=theme.ACCENT,
        )
        self._state_lbl.configure(text="Starting camera stream…")
        self._footer_lbl.configure(text=f"Camera {cams_text} live feed")
        self._reset_display()
        self._canvas.delete("all")
        cw = self._canvas.winfo_width() or 600
        ch = self._canvas.winfo_height() or 400
        self._canvas.create_text(
            cw // 2, ch // 2,
            text="Opening camera feed — press  ⏹ Stop  to end",
            fill=theme.TEXT_3, font=("Segoe UI", 12),
        )

        # Set lane mode automatically based on number of cameras
        lane_label = f"{num_cams} lane" if num_cams == 1 else f"{num_cams} lanes"
        if num_cams in (1, 2, 4):
            self._lane_mode.set(lane_label)
            self._on_lane_mode_change(lane_label)
        else:
            self.detector.set_lane_count(num_cams)

        self.detector.grayscale_mode = bool(self._gray_chk.get())
        self._lock_inputs(True)
        self._update_device_ui()
        self._stop_btn.configure(state="normal")
        self._pause_btn.configure(state="normal")
        self._resume_btn.configure(state="disabled")

        logger.info(f"Starting live camera detection on camera index list: {cam_indices}.")
        self.detector.start(
            cam_indices,
            save_output=False,   # live feeds — no file output
            is_camera=True,
        )

    def _on_grayscale_toggle(self):
        """Toggle grayscale video mode dynamically."""
        enabled = bool(self._gray_chk.get())
        self.detector.grayscale_mode = enabled
        logger.info(f"Grayscale display mode: {enabled}")

    def _on_conf_slider_change(self, value):
        """Dynamically update confidence threshold from GUI slider."""
        val_int = int(value)
        if hasattr(self, "_conf_val_lbl"):
            self._conf_val_lbl.configure(text=f"Min Confidence: {val_int}%")
        self.detector.confidence = val_int / 100.0

    def open_output_folder(self):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        try:
            os.startfile(config.OUTPUT_DIR)
        except Exception:
            self._dialog("Error", "Could not open the output folder.")

    # ─────────────────────────────────────────────────────────────────────────
    # Detector event queue polling
    # ─────────────────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                evt, data = self.detector.event_queue.get_nowait()
                self._on_event(evt, data)
                self.detector.event_queue.task_done()
        except queue.Empty:
            pass
        self._redraw()
        # Throttle display work to a predictable rate while the detector keeps
        # publishing the newest frame. This reduces Tk/PIL overhead and keeps
        # live camera latency from growing on slower machines.
        poll_ms = max(1, round(1000 / config.GUI_RENDER_FPS_CAP)) if self._camera_active else 20
        self.after(poll_ms, self._poll)

    def _on_event(self, evt: str, data):
        if evt == "state_change":
            self._state_lbl.configure(text=data)
            self._footer_lbl.configure(text=data)
            if data == "Finished":
                self.stop_detection()
                self._progress.set(1.0)
                self._pct_lbl.configure(text="100%")
                self._dialog("Complete", "Detection finished.\nOutput video saved to the output folder.")
            elif data == "Camera Stopped":
                # Camera feed ended (user stopped or camera disconnected)
                self._camera_active = False
                self.stop_detection()
                self._state_lbl.configure(text="Camera stopped")
                # Restore the source label
                self._source_lbl.configure(text="Camera stopped", text_color=theme.TEXT_2)

        elif evt == "stats_update":
            self.detector.mark_stats_event_processed()
            self._apply_stats(self.detector.stats)

        elif evt == "error":
            self.stop_detection()
            self._state_lbl.configure(text="Error")
            self._dialog("Runtime Error", str(data))

    def _apply_stats(self, s):
        self._progress.set(s.progress_percent / 100)
        self._pct_lbl.configure(text=f"{s.progress_percent:.1f}%")
        self._perf["fps"].configure(text=f"{s.fps:.1f} fps")
        self._perf["inf"].configure(text=f"{s.inference_time_ms:.1f} ms")
        self._perf["frame"].configure(text=f"{s.frame_number}/{s.total_frames}")
        self._perf["time"].configure(text=utils.format_time(s.elapsed_time))
        self._perf["res"].configure(text=s.resolution)
        self._cur_lbl.configure(text=str(s.current_count))
        self._tot_lbl.configure(text=str(s.total_detected))

        # Update Traffic Congestion Density Meter
        max_vehicles = max(s.lane_vehicle_counts.values()) if s.lane_vehicle_counts else 0
        if max_vehicles < config.CONGESTION_LOW:
            c_text, c_color, c_pct = "Traffic: Low Density 🟢", theme.SUCCESS, max(0.08, max_vehicles / 15.0)
        elif max_vehicles < config.CONGESTION_MEDIUM:
            c_text, c_color, c_pct = "Traffic: Moderate 🟡", theme.WARNING, min(0.50, max_vehicles / 25.0)
        elif max_vehicles < config.CONGESTION_HIGH:
            c_text, c_color, c_pct = "Traffic: Heavy 🟠", "#f97316", min(0.85, max_vehicles / 35.0)
        else:
            c_text, c_color, c_pct = "Traffic: CRITICAL CONGESTION 🔴", theme.DANGER, 1.0

        if hasattr(self, "_congestion_lbl"):
            self._congestion_lbl.configure(text=c_text, text_color=c_color)
            self._congestion_bar.configure(progress_color=c_color)
            self._congestion_bar.set(c_pct)
        lane_text = "  ".join(
            f"Lane {lane}: {count}"
            for lane, count in sorted(s.lane_vehicle_counts.items())
        )
        self._lane_counts_val.configure(text=lane_text or "Lane counts unavailable")
        signal_color = theme.DANGER if s.ambulance_lanes else theme.SUCCESS
        self._signal_val.configure(text=s.signal_advice, text_color=signal_color)
        emergency_color = theme.DANGER if s.ambulance_lanes else theme.TEXT_3
        self._emergency_val.configure(text=s.emergency_advice, text_color=emergency_color)
        self._sync_classes(s.class_counts_current, s.class_counts_total)

        # ── Emergency banner ──────────────────────────────────────────────
        if s.ambulance_lanes:
            if not self._emergency_banner_visible:
                self._emergency_banner.grid()   # make visible
                self._emergency_banner_visible = True
            self._emergency_banner.show(
                lanes=s.ambulance_lanes,
                directions=getattr(s, 'ambulance_directions', {}),
            )
        else:
            if self._emergency_banner_visible:
                self._emergency_banner.hide()
                self._emergency_banner.grid_remove()
                self._emergency_banner_visible = False

        # ── Per-lane signal timer rows ─────────────────────────────────────
        for lane, row in self._signal_timer_rows.items():
            state    = s.lane_signal_states.get(lane, "RED")
            g_dur    = s.green_durations.get(lane, 0)
            veh_cnt  = s.lane_vehicle_counts.get(lane, 0)
            row.update_state(state, g_dur, veh_cnt)

    def _sync_classes(self, current: dict, total: dict):
        all_cls = sorted(set(list(current.keys()) + list(total.keys())))
        for cls in all_cls:
            cur = current.get(cls, 0)
            tot = total.get(cls, 0)
            if cls not in self._cls_widgets:
                row = ctk.CTkFrame(self._cls_frame, fg_color="transparent")
                row.pack(fill="x", pady=1)
                color = self.detector.color_hex(cls)
                ctk.CTkLabel(
                    row, text="●",
                    font=("Segoe UI", 10), text_color=color
                ).pack(side="left", padx=(0, 6))
                ctk.CTkLabel(
                    row, text=cls,
                    font=theme.label(), text_color=theme.TEXT_1
                ).pack(side="left")
                val = ctk.CTkLabel(
                    row, text=f"{cur} / {tot}",
                    font=theme.mono(), text_color=theme.TEXT_2
                )
                val.pack(side="right")
                self._cls_widgets[cls] = (row, val)
            else:
                _, val = self._cls_widgets[cls]
                val.configure(text=f"{cur} / {tot}")

    # ─────────────────────────────────────────────────────────────────────────
    # Canvas rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _redraw(self):
        """Pull the latest processed frame from the detector and paint it."""
        try:
            sequence, frame = self.detector.get_latest_frame_if_new(self._last_frame_sequence)
            if frame is None:
                return
            self._last_frame_sequence = sequence

            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            if cw < 4 or ch < 4:
                return

            img = Image.fromarray(frame)
            fw, fh = img.size
            ratio = min(cw / fw, ch / fh)
            resampling = (
                Image.Resampling.BILINEAR
                if self._camera_active else Image.Resampling.LANCZOS
            )
            img = img.resize(
                (max(1, int(fw * ratio)), max(1, int(fh * ratio))),
                resampling
            )
            photo = ImageTk.PhotoImage(img)
            self._canvas.delete("all")
            self._canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
            self._canvas.image = photo  # hold reference against GC
        except (tk.TclError, Exception):
            pass

        # Mirror to fullscreen overlay if open
        if self._fullscreen_win and self._fullscreen_lbl:
            try:
                lw = self._fullscreen_lbl.winfo_width()
                lh = self._fullscreen_lbl.winfo_height()
                if lw > 4 and lh > 4:
                    r2 = min(lw / fw, lh / fh)
                    fs = Image.fromarray(frame).resize(
                        (max(1, int(fw * r2)), max(1, int(fh * r2))),
                        resampling
                    )
                    fp = ImageTk.PhotoImage(fs)
                    self._fullscreen_lbl.configure(image=fp)
                    self._fullscreen_lbl.image = fp
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Fullscreen overlay
    # ─────────────────────────────────────────────────────────────────────────

    def open_fullscreen(self, event=None):
        if self.detector.get_latest_frame() is None:
            return
        if self._fullscreen_win:
            return
        self._fullscreen_win = ctk.CTkToplevel(self)
        self._fullscreen_win.title("Fullscreen")
        self._fullscreen_win.attributes("-fullscreen", True)
        self._fullscreen_win.configure(fg_color=theme.BG_ROOT)
        self._fullscreen_win.bind("<Escape>", self.close_fullscreen)

        self._fullscreen_lbl = ctk.CTkLabel(
            self._fullscreen_win, text="", fg_color=theme.BG_ROOT
        )
        self._fullscreen_lbl.pack(fill="both", expand=True)
        self._fullscreen_lbl.bind("<Double-Button-1>", self.close_fullscreen)
        self._fullscreen_win.focus_set()

    def close_fullscreen(self, event=None):
        if self._fullscreen_win:
            self._fullscreen_win.destroy()
            self._fullscreen_win = None
            self._fullscreen_lbl = None

    # ─────────────────────────────────────────────────────────────────────────
    # History
    # ─────────────────────────────────────────────────────────────────────────

    def _add_history(self, path: str):
        if path in self._history:
            self._history.remove(path)
        self._history.insert(0, path)
        if len(self._history) > 10:
            self._history.pop()
        self._save_history()
        self._refresh_history()

    def _load_history(self):
        if os.path.exists(self._history_file):
            try:
                with open(self._history_file, encoding="utf-8") as fh:
                    self._history = [l.strip() for l in fh if l.strip()]
                self._refresh_history()
            except Exception:
                pass

    def _save_history(self):
        try:
            with open(self._history_file, "w", encoding="utf-8") as fh:
                fh.writelines(f"{p}\n" for p in self._history)
        except Exception:
            pass

    def _refresh_history(self):
        self._hist_lb.delete(0, tk.END)
        for p in self._history:
            self._hist_lb.insert(tk.END, f"  {os.path.basename(p)}")

    def _hist_click(self, _event):
        sel = self._hist_lb.curselection()
        if not sel:
            return
        path = self._history[sel[0]]
        if os.path.exists(path):
            self.set_lane_video(0, path)
        else:
            self._dialog("File Not Found", f"File no longer exists:\n{path}")
            self._history.pop(sel[0])
            self._save_history()
            self._refresh_history()

    # ─────────────────────────────────────────────────────────────────────────
    # Per-lane YouTube download
    # ─────────────────────────────────────────────────────────────────────────

    def _start_lane_yt_download(self, lane_index: int):
        """Open a small dialog to paste a YouTube URL for a specific lane."""
        dlg = ctk.CTkToplevel(self)
        dlg.title(f"YouTube Download — Lane {lane_index + 1}")
        dlg.geometry("420x160")
        dlg.resizable(False, False)
        dlg.configure(fg_color=theme.BG_SURFACE)
        dlg.transient(self)
        dlg.grab_set()
        dlg.update_idletasks()
        x = self.winfo_x() + self.winfo_width() // 2 - 210
        y = self.winfo_y() + self.winfo_height() // 2 - 80
        dlg.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            dlg, text=f"YouTube URL for Lane {lane_index + 1}",
            font=theme.label_bold(), text_color=theme.ACCENT
        ).pack(pady=(18, 6))

        url_entry = ctk.CTkEntry(
            dlg,
            placeholder_text="Paste YouTube URL…",
            fg_color=theme.BG_ELEVATED,
            border_color=theme.BORDER, border_width=1,
            text_color=theme.TEXT_1,
            placeholder_text_color=theme.TEXT_3,
            corner_radius=theme.R_MD,
            font=theme.label(), height=32, width=360,
        )
        url_entry.pack(padx=20, pady=(0, 10))

        def _do_download():
            url = url_entry.get().strip()
            if not url:
                return
            dlg.destroy()
            if not utils.check_internet_connection():
                self._dialog("No Network", "Cannot reach the internet.")
                return

            self._lane_slot_lbls[lane_index].configure(
                text=f"Lane {lane_index + 1}: downloading…",
                text_color=theme.WARNING,
            )
            if lane_index < len(self._lane_yt_btns):
                self._lane_yt_btns[lane_index].configure(state="disabled")
            if lane_index < len(self._lane_slot_btns):
                self._lane_slot_btns[lane_index].configure(state="disabled")

            downloader = YouTubeDownloader()
            self._lane_yt_downloaders[lane_index] = downloader

            def _worker():
                try:
                    path = downloader.download(
                        url, config.OUTPUT_DIR,
                        lambda pct, spd, eta, _: self.after(
                            0, lambda p=pct, li=lane_index:
                            self._lane_slot_lbls[li].configure(
                                text=f"Lane {li+1}: {p:.0f}%",
                                text_color=theme.WARNING,
                            )
                        ),
                    )
                    self.after(0, lambda p=path, li=lane_index: self._lane_yt_done(li, p))
                except Exception as exc:
                    self.after(0, lambda li=lane_index, e=exc: self._lane_yt_error(li, e))

            t = threading.Thread(target=_worker, name=f"LaneYT-{lane_index}", daemon=True)
            self._lane_yt_threads[lane_index] = t
            t.start()

        ctk.CTkButton(
            dlg, text="Download & Assign",
            font=theme.label_bold(),
            fg_color=theme.ACCENT, hover_color=theme.ACCENT_DIM,
            corner_radius=theme.R_MD, height=32,
            command=_do_download,
        ).pack()

    def _lane_yt_done(self, lane_index: int, path: str):
        self.set_lane_video(lane_index, path)
        if lane_index < len(self._lane_yt_btns):
            self._lane_yt_btns[lane_index].configure(state="normal")
        if lane_index < len(self._lane_slot_btns):
            self._lane_slot_btns[lane_index].configure(state="normal")

    def _lane_yt_error(self, lane_index: int, exc: Exception):
        self._lane_slot_lbls[lane_index].configure(
            text=f"Lane {lane_index + 1}: download failed",
            text_color=theme.DANGER,
        )
        if lane_index < len(self._lane_yt_btns):
            self._lane_yt_btns[lane_index].configure(state="normal")
        if lane_index < len(self._lane_slot_btns):
            self._lane_slot_btns[lane_index].configure(state="normal")
        self._dialog("Download Failed", f"Lane {lane_index + 1} download error:\n{exc}")

    def clear_history(self):
        self._history.clear()
        self._hist_lb.delete(0, tk.END)
        try:
            if os.path.exists(self._history_file):
                os.remove(self._history_file)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_display(self):
        """Reset progress, stats labels, and dynamic class rows."""
        self._cur_lbl.configure(text="0")
        self._tot_lbl.configure(text="0")
        self._progress.set(0)
        self._pct_lbl.configure(text="0%")
        for lbl in self._perf.values():
            lbl.configure(text="—")
        for row, _ in self._cls_widgets.values():
            row.destroy()
        self._cls_widgets.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluate Model dialog
    # ─────────────────────────────────────────────────────────────────────────

    def _load_yaml_cache(self) -> str:
        """Returns the last-used data.yaml path, or '' if none cached."""
        try:
            if os.path.isfile(config.EVAL_YAML_CACHE):
                with open(config.EVAL_YAML_CACHE, encoding="utf-8") as fh:
                    path = fh.read().strip()
                if os.path.isfile(path):
                    return path
        except Exception:
            pass
        return config.DEFAULT_EVAL_YAML if os.path.isfile(config.DEFAULT_EVAL_YAML) else ""

    def _save_yaml_cache(self, path: str) -> None:
        """Persists the selected data.yaml path for future sessions."""
        try:
            with open(config.EVAL_YAML_CACHE, "w", encoding="utf-8") as fh:
                fh.write(path)
        except Exception:
            pass

    def open_eval_dialog(self):
        """Opens the Evaluate Model dialog window."""
        if self._eval_thread and self._eval_thread.is_alive():
            self._dialog("Evaluation Running", "An evaluation is already in progress.")
            return

        dlg = ctk.CTkToplevel(self)
        dlg.title("Evaluate Model  ·  Performance Metrics")
        dlg.geometry("560x820")
        dlg.minsize(480, 740)
        dlg.configure(fg_color=theme.BG_SURFACE)
        dlg.transient(self)
        dlg.grab_set()
        dlg.update_idletasks()

        # Centre relative to main window
        x = self.winfo_x() + self.winfo_width()  // 2 - 280
        y = self.winfo_y() + self.winfo_height() // 2 - 380
        dlg.geometry(f"+{x}+{y}")

        # ── Header ─────────────────────────────────────────────────────────
        ctk.CTkLabel(
            dlg, text="📊  MODEL EVALUATION",
            font=theme.title(), text_color=theme.ACCENT
        ).pack(pady=(16, 2))

        eval_info_box = ctk.CTkFrame(dlg, fg_color=theme.BG_ELEVATED, corner_radius=theme.R_MD, border_color=theme.BORDER, border_width=1)
        eval_info_box.pack(fill="x", padx=20, pady=(6, 12))

        ctk.CTkLabel(
            eval_info_box,
            text="Evaluation mode",
            font=theme.caption(), text_color=theme.TEXT_3
        ).pack(pady=(6, 2))
        evaluation_mode = tk.StringVar(value="Standard validation")
        ctk.CTkOptionMenu(
            eval_info_box,
            values=["Standard validation", "Grayscale robustness"],
            variable=evaluation_mode,
            font=theme.label_bold(),
            fg_color=theme.BG_SURFACE,
            button_color=theme.ACCENT,
            button_hover_color=theme.ACCENT_DIM,
            dropdown_fg_color=theme.BG_ELEVATED,
            dropdown_hover_color=theme.BORDER,
            width=210, height=30
        ).pack(pady=(0, 6))

        ctk.CTkFrame(dlg, height=1, fg_color=theme.BORDER).pack(fill="x", padx=20, pady=(0, 10))

        # ── YAML path row ──────────────────────────────────────────────────
        yaml_label = ctk.CTkLabel(
            dlg, text="data.yaml  (required for evaluation)",
            font=theme.label(), text_color=theme.TEXT_2, anchor="w"
        )
        yaml_label.pack(fill="x", padx=20)

        yaml_row = ctk.CTkFrame(dlg, fg_color="transparent")
        yaml_row.pack(fill="x", padx=20, pady=(4, 0))
        yaml_row.grid_columnconfigure(0, weight=1)

        yaml_var = tk.StringVar(value=self._load_yaml_cache())

        yaml_entry = ctk.CTkEntry(
            yaml_row,
            textvariable=yaml_var,
            placeholder_text="Select data.yaml…",
            fg_color=theme.BG_ELEVATED,
            border_color=theme.BORDER, border_width=1,
            text_color=theme.TEXT_1,
            placeholder_text_color=theme.TEXT_3,
            corner_radius=theme.R_MD,
            font=theme.label(), height=32
        )
        yaml_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        def _browse_yaml():
            path = filedialog.askopenfilename(
                title="Select data.yaml",
                filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
            )
            if path:
                yaml_var.set(os.path.normpath(path))

        ctk.CTkButton(
            yaml_row, text="Browse",
            font=theme.label(),
            fg_color="transparent", text_color=theme.TEXT_1,
            border_color=theme.BORDER, border_width=1,
            hover_color=theme.BG_ELEVATED,
            corner_radius=theme.R_MD, height=32, width=72,
            command=_browse_yaml
        ).grid(row=0, column=1)

        # ── Model path row ─────────────────────────────────────────────────
        ctk.CTkFrame(dlg, height=1, fg_color=theme.BORDER).pack(fill="x", padx=20, pady=(14, 10))

        model_info = ctk.CTkLabel(
            dlg,
            text=f"Model: {os.path.basename(config.MODEL_PATH)}",
            font=theme.label(), text_color=theme.TEXT_2, anchor="w"
        )
        model_info.pack(fill="x", padx=20, pady=(0, 4))

        _, dev_name = utils.get_device_info(self.detector.device)
        ctk.CTkLabel(
            dlg,
            text=f"Device: {dev_name}",
            font=theme.caption(), text_color=theme.TEXT_3, anchor="w"
        ).pack(fill="x", padx=20, pady=(0, 14))

        ctk.CTkFrame(dlg, height=1, fg_color=theme.BORDER).pack(fill="x", padx=20, pady=(0, 12))

        # ── Status / progress area ─────────────────────────────────────────
        # Optional custom PNG charts. Values are saved beside the selected
        # evaluation results and can be generated before or after evaluation.
        chart_box = ctk.CTkFrame(
            dlg, fg_color=theme.BG_ELEVATED, corner_radius=theme.R_MD,
            border_color=theme.BORDER, border_width=1,
        )
        chart_box.pack(fill="x", padx=20, pady=(0, 12))
        ctk.CTkLabel(
            chart_box, text="Custom curve and confusion matrix (PNG)",
            font=theme.label_bold(), text_color=theme.TEXT_1, anchor="w",
        ).pack(fill="x", padx=12, pady=(9, 1))
        ctk.CTkLabel(
            chart_box,
            text="Curve values must be 0-1; separate matrix rows with semicolons.",
            font=theme.caption(), text_color=theme.TEXT_3, anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 7))

        curve_var = tk.StringVar(value="0.20, 0.35, 0.48, 0.62, 0.75, 0.86, 0.92")
        matrix_var = tk.StringVar(value="80,10,5; 8,85,7; 4,6,90")
        labels_var = tk.StringVar(value="car, bus, truck")
        chart_box.grid_columnconfigure(1, weight=1)

        for row, label, variable, placeholder in (
            (0, "Curve values", curve_var, "0.2, 0.4, 0.7"),
            (1, "Matrix rows", matrix_var, "80,10; 8,90"),
            (2, "Class labels", labels_var, "car, bus, truck"),
        ):
            ctk.CTkLabel(
                chart_box, text=label, font=theme.caption(),
                text_color=theme.TEXT_2, anchor="w",
            ).grid(row=row, column=0, padx=(12, 8), pady=3, sticky="w")
            ctk.CTkEntry(
                chart_box, textvariable=variable, placeholder_text=placeholder,
                font=theme.caption(), fg_color=theme.BG_SURFACE,
                border_color=theme.BORDER, height=28,
            ).grid(row=row, column=1, padx=(0, 12), pady=3, sticky="ew")

        chart_status_var = tk.StringVar(value="")

        def _generate_charts():
            """Validate chart fields and save both images in the active folder."""
            output_dir = (
                config.EVAL_RESULTS_DIR
                if evaluation_mode.get() == "Standard validation"
                else config.ROBUSTNESS_RESULTS_GRAYSCALE_DIR
            )
            try:
                paths = generate_custom_evaluation_charts(
                    curve_values=curve_var.get(),
                    matrix_values=matrix_var.get(),
                    class_labels=labels_var.get(),
                    output_dir=output_dir,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                chart_status_var.set(f"Chart error: {exc}")
                return
            chart_status_var.set(
                "Saved "
                f"{os.path.basename(paths['curve'])} and "
                f"{os.path.basename(paths['matrix'])}"
            )
            open_folder_btn.configure(state="normal")
            logger.info("Custom evaluation charts saved to %s", output_dir)

        chart_btn = ctk.CTkButton(
            chart_box, text="Generate Curve & Matrix PNGs",
            font=theme.label_bold(), fg_color=theme.ACCENT,
            hover_color=theme.ACCENT_DIM, corner_radius=theme.R_MD,
            height=30, command=_generate_charts,
        )
        chart_btn.grid(row=3, column=0, columnspan=2, padx=12, pady=(7, 2), sticky="ew")
        ctk.CTkLabel(
            chart_box, textvariable=chart_status_var,
            font=theme.caption(), text_color=theme.TEXT_3, anchor="w",
        ).grid(row=4, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="w")

        status_var = tk.StringVar(value="Ready to evaluate.")

        status_lbl = ctk.CTkLabel(
            dlg, textvariable=status_var,
            font=theme.caption(), text_color=theme.TEXT_3, anchor="w"
        )
        status_lbl.pack(fill="x", padx=20, pady=(0, 4))

        progress_bar = ctk.CTkProgressBar(
            dlg,
            fg_color=theme.BG_ELEVATED,
            progress_color=theme.ACCENT,
            corner_radius=1, height=3
        )
        progress_bar.pack(fill="x", padx=20, pady=(0, 14))
        progress_bar.set(0)

        # ── Results text area ──────────────────────────────────────────────
        results_box = ctk.CTkTextbox(
            dlg,
            fg_color=theme.BG_ELEVATED,
            border_color=theme.BORDER, border_width=1,
            text_color=theme.TEXT_1,
            font=theme.mono(),
            corner_radius=theme.R_MD,
            state="disabled",
            wrap="none"
        )
        results_box.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # ── Bottom buttons ─────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 18))

        run_btn = ctk.CTkButton(
            btn_row, text="▶  Run Evaluation",
            font=theme.label_bold(),
            fg_color=theme.SUCCESS, hover_color="#16a34a",
            corner_radius=theme.R_MD, height=34,
        )
        run_btn.pack(side="left", padx=(0, 8))

        def _open_results_folder():
            folder = (
                config.EVAL_RESULTS_DIR
                if evaluation_mode.get() == "Standard validation"
                else config.ROBUSTNESS_RESULTS_GRAYSCALE_DIR
            )
            os.makedirs(folder, exist_ok=True)
            os.startfile(folder)

        open_folder_btn = ctk.CTkButton(
            btn_row, text="Open Results Folder",
            font=theme.caption(),
            fg_color="transparent", text_color=theme.TEXT_2,
            hover_color=theme.BG_ELEVATED,
            border_width=0,
            corner_radius=theme.R_MD, height=28,
            state="disabled",
            command=_open_results_folder,
        )
        open_folder_btn.pack(side="left")


        ctk.CTkButton(
            btn_row, text="Close",
            font=theme.label(),
            fg_color="transparent", text_color=theme.TEXT_2,
            hover_color=theme.BG_ELEVATED,
            border_color=theme.BORDER, border_width=1,
            corner_radius=theme.R_MD, height=34, width=72,
            command=dlg.destroy
        ).pack(side="right")

        # ── Internal helpers ───────────────────────────────────────────────

        def _set_results_text(text: str):
            results_box.configure(state="normal")
            results_box.delete("1.0", "end")
            results_box.insert("1.0", text)
            results_box.configure(state="disabled")

        def _on_progress(msg: str):
            """Called from the evaluation thread — must schedule GUI updates via after()."""
            dlg.after(0, lambda: status_var.set(msg))

        def _on_eval_done(result: EvaluationResult, mode: str):
            """Called on the main thread when evaluation finishes successfully."""
            progress_bar.stop()
            progress_bar.configure(mode="determinate")
            progress_bar.set(1.0)
            run_btn.configure(state="normal", text="▶  Run Evaluation")
            self._eval_btn.configure(state="normal")
            chart_btn.configure(state="normal")
            open_folder_btn.configure(state="normal")

            elapsed = f"{result.elapsed_sec:.1f}s"
            lines = [
                "=" * 48,
                f"  {mode.upper()} RESULTS",
                "=" * 48,
                "",
                f"  {'Precision':<18} {result.precision:.4f}",
                f"  {'Recall':<18} {result.recall:.4f}",
                f"  {'mAP@0.50':<18} {result.map50:.4f}",
                f"  {'mAP@0.50:0.95':<18} {result.map5095:.4f}",
                f"  {'F1-Score':<18} {result.f1:.4f}",
                "",
                "-" * 48,
                "  Per-class  (Prec / Recall / AP50)",
                "-" * 48,
            ]
            for cls, vals in result.per_class.items():
                lines.append(
                    f"  {cls:<18} {vals['p']:.3f} / {vals['r']:.3f} / {vals['ap50']:.3f}"
                )
            lines += [
                "",
                "-" * 48,
                f"  Elapsed: {elapsed}",
                f"  Saved:   {result.metrics_txt}",
                "=" * 48,
            ]
            _set_results_text("\n".join(lines))
            output_name = (
                "evaluation_results/" if mode == "Standard validation"
                else "robustness_eval/results/grayscale/"
            )
            status_var.set(f"Done  ({elapsed})  Results saved to {output_name}")
            logger.info(f"Evaluation complete. Precision={result.precision} mAP50={result.map50}")

        def _on_eval_error(msg: str):
            """Called on the main thread when evaluation fails."""
            progress_bar.stop()
            progress_bar.configure(mode="determinate")
            progress_bar.set(0)
            run_btn.configure(state="normal", text="▶  Run Evaluation")
            self._eval_btn.configure(state="normal")
            chart_btn.configure(state="normal")
            status_var.set("Evaluation failed.")
            _set_results_text(f"ERROR\n\n{msg}\n\nCheck logs/detector.log for details.")
            logger.error(f"Evaluation failed: {msg}")

        def _run_eval(mode: str):
            """Worker function executed on a daemon background thread."""
            yaml_path = yaml_var.get().strip()
            if not yaml_path or not os.path.isfile(yaml_path):
                dlg.after(0, lambda: _on_eval_error(
                    "data.yaml file not found.\n\nPlease click Browse and select a valid data.yaml file."
                ))
                return

            self._save_yaml_cache(yaml_path)

            if mode == "Standard validation":
                result = run_standard_evaluation(
                    model_path=config.MODEL_PATH,
                    yaml_path=yaml_path,
                    output_dir=config.EVAL_RESULTS_DIR,
                    device=self.detector.device,
                    progress_cb=_on_progress,
                )
            else:
                result = run_evaluation(
                    model_path=config.MODEL_PATH,
                    yaml_path=yaml_path,
                    output_dir=config.ROBUSTNESS_RESULTS_GRAYSCALE_DIR,
                    device=self.detector.device,
                    progress_cb=_on_progress,
                )

            if result.success:
                dlg.after(0, lambda r=result, m=mode: _on_eval_done(r, m))
            else:
                dlg.after(0, lambda: _on_eval_error(result.error))

        def _start_eval():
            """Triggered by the Run Evaluation button click."""
            yaml_path = yaml_var.get().strip()
            if not yaml_path:
                status_var.set("Please select a data.yaml file first.")
                return

            if not self.model_loaded:
                self._dialog("No Model", "weights/best.pt is not loaded.\nCannot run evaluation.")
                return

            # Lock the button and start the spinner
            run_btn.configure(state="disabled", text="Running…")
            self._eval_btn.configure(state="disabled")
            chart_btn.configure(state="disabled")
            open_folder_btn.configure(state="disabled")
            _set_results_text("")
            status_var.set("Starting evaluation…")

            progress_bar.configure(mode="indeterminate")
            progress_bar.start()
            mode = evaluation_mode.get()

            self._eval_thread = threading.Thread(
                target=_run_eval,
                args=(mode,),
                name="EvaluationWorker",
                daemon=True
            )
            self._eval_thread.start()

        run_btn.configure(command=_start_eval)

    # ─────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────────────

    def _on_close(self):
        logger.info("Shutting down.")
        self.stop_detection()
        self.destroy()
