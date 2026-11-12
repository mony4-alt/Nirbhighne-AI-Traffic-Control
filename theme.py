"""theme.py — Minimalistic dark colour and font system."""
import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Backgrounds ───────────────────────────────────────────────────────────────
BG_ROOT     = "#0c0c0c"   # Deep background (window, canvas)
BG_SURFACE  = "#111111"   # Sidebar / panel surfaces
BG_ELEVATED = "#1a1a1a"   # Inputs, cards, hover targets

# ── Borders ───────────────────────────────────────────────────────────────────
BORDER      = "#262626"   # Subtle 1-px dividers

# ── Accent ────────────────────────────────────────────────────────────────────
ACCENT      = "#818cf8"   # Indigo-400  (primary interactive)
ACCENT_DIM  = "#6366f1"   # Indigo-500  (hover / pressed)

# ── Status / functional colours ───────────────────────────────────────────────
SUCCESS     = "#22c55e"   # Green  — Start button, total count
DANGER      = "#ef4444"   # Red    — Stop button, clear link
WARNING     = "#f59e0b"   # Amber  — Pause button, device unknown

# ── Text ──────────────────────────────────────────────────────────────────────
TEXT_1      = "#ffffff"   # Primary readable text
TEXT_2      = "#d1d5db"   # Secondary labels, stat row titles
TEXT_3      = "#a3a3a3"   # Muted / placeholder / section headers

# ── Radii ─────────────────────────────────────────────────────────────────────
R_SM = 4
R_MD = 8
R_LG = 12

# ── Fonts ─────────────────────────────────────────────────────────────────────
def title():         return ctk.CTkFont("Segoe UI", 14, "bold")
def heading():       return ctk.CTkFont("Segoe UI", 12, "bold")
def label():         return ctk.CTkFont("Segoe UI", 11)
def label_bold():    return ctk.CTkFont("Segoe UI", 11, "bold")
def caption():       return ctk.CTkFont("Segoe UI", 9)
def caption_bold():  return ctk.CTkFont("Segoe UI", 9, "bold")
def mono():          return ctk.CTkFont("Consolas", 11)
def mono_lg():       return ctk.CTkFont("Consolas", 22, "bold")
