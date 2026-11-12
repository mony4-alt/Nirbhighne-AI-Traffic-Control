"""main.py — Application entry point with minimal splash screen."""
import os
import sys

# Ensure the project directory is on sys.path when frozen or run directly
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils import fix_pyinstaller_streams, logger
fix_pyinstaller_streams()

from PIL import Image
import customtkinter as ctk
import theme
import config
from gui import AppGUI


class SplashScreen(ctk.CTk):
    """Minimal borderless startup splash with application logo."""

    _STAGES = [
        (0.20, "Reading environment…"),
        (0.45, "Checking GPU / CUDA…"),
        (0.70, "Locating model weights…"),
        (0.90, "Initialising neural network…"),
        (1.00, "Opening application…"),
    ]

    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.configure(fg_color=theme.BG_SURFACE)

        if os.path.exists(config.APP_ICON_ICO):
            try:
                self.iconbitmap(config.APP_ICON_ICO)
            except Exception:
                pass

        w, h = 440, 230
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        # Logo Image
        if os.path.exists(config.APP_ICON_PNG):
            try:
                pil_img = Image.open(config.APP_ICON_PNG)
                self._logo_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(54, 54))
                logo_lbl = ctk.CTkLabel(self, image=self._logo_img, text="")
                logo_lbl.pack(pady=(18, 0))
            except Exception:
                pass

        # App name
        ctk.CTkLabel(
            self, text="NIRBHIGHNE",
            font=ctk.CTkFont("Segoe UI", 22, "bold"),
            text_color=theme.ACCENT
        ).pack(pady=(4, 2))

        ctk.CTkLabel(
            self,
            text="AI-Driven Dynamic Traffic Signal Controlling System",
            font=ctk.CTkFont("Segoe UI", 9),
            text_color=theme.TEXT_2
        ).pack()

        self._status_lbl = ctk.CTkLabel(
            self, text="Initialising…",
            font=ctk.CTkFont("Consolas", 9),
            text_color=theme.TEXT_3
        )
        self._status_lbl.pack(pady=(10, 6))


        self._bar = ctk.CTkProgressBar(
            self,
            fg_color=theme.BG_ELEVATED,
            progress_color=theme.ACCENT,
            corner_radius=1,
            height=3,
            width=340
        )
        self._bar.pack()
        self._bar.set(0.0)

        self._val = 0.0
        self._tick()

    def _tick(self):
        """Advance progress bar and update stage label."""
        if self._val < 1.0:
            self._val = min(1.0, self._val + 0.035)
            self._bar.set(self._val)
            for threshold, msg in self._STAGES:
                if self._val <= threshold:
                    self._status_lbl.configure(text=msg)
                    break
            self.after(55, self._tick)
        else:
            self.destroy()


def main():
    logger.info("Nirbhighne starting.")
    splash = SplashScreen()
    splash.mainloop()

    logger.info("Launching main application window.")
    app = AppGUI()
    app.mainloop()
    logger.info("Session terminated.")


if __name__ == "__main__":
    main()

