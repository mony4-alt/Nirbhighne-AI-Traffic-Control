"""emergency_overlay.py — Reusable GUI emergency-alert banner widget.

Displayed above the video canvas when an ambulance is confirmed in any lane.
The banner flashes with a red/amber background, shows the lane number,
direction arrow, and a "YIELD / CLEAR LANE" message.
"""
from __future__ import annotations

import tkinter as tk
import customtkinter as ctk
import theme


class EmergencyBanner(ctk.CTkFrame):
    """Flashing emergency banner shown when an ambulance is detected.

    Usage
    ─────
        banner = EmergencyBanner(parent)
        banner.pack(...)
        banner.show(lanes=[1, 2], directions={1: 'left', 2: 'unknown'})
        banner.hide()
    """

    _FLASH_COLORS = ("#c00000", "#7a0000")   # bright red ↔ dark red
    _FLASH_INTERVAL_MS = 400                  # ms between colour flips

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            fg_color=self._FLASH_COLORS[0],
            corner_radius=0,
            height=60,
            **kwargs,
        )
        self.pack_propagate(False)
        self._visible = False
        self._flash_idx = 0
        self._flash_job: str | None = None

        # ── Layout ────────────────────────────────────────────────────────
        self._icon_lbl = ctk.CTkLabel(
            self,
            text="🚑",
            font=ctk.CTkFont("Segoe UI Emoji", 28),
            fg_color="transparent",
            text_color="#ffffff",
        )
        self._icon_lbl.pack(side="left", padx=(14, 0))

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(side="left", fill="both", expand=True, padx=8)

        self._title_lbl = ctk.CTkLabel(
            inner,
            text="AMBULANCE DETECTED — EMERGENCY",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            text_color="#ffffff",
            anchor="w",
        )
        self._title_lbl.pack(fill="x")

        self._sub_lbl = ctk.CTkLabel(
            inner,
            text="",
            font=ctk.CTkFont("Segoe UI", 11),
            text_color="#ffcccc",
            anchor="w",
        )
        self._sub_lbl.pack(fill="x")

        # Right-side "YIELD" badge
        badge = ctk.CTkFrame(
            self, fg_color="#800000", corner_radius=6, width=70
        )
        badge.pack(side="right", padx=14, pady=10)
        badge.pack_propagate(False)
        ctk.CTkLabel(
            badge,
            text="YIELD\nNOW",
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
            text_color="#ffffff",
            justify="center",
        ).pack(expand=True)

    # ── Public API ────────────────────────────────────────────────────────

    def show(
        self,
        lanes: list[int],
        directions: dict[int, str] | None = None,
    ):
        """Make the banner visible and start flashing."""
        directions = directions or {}
        self._visible = True

        lane_str = f"Lane{'s' if len(lanes) > 1 else ''} {', '.join(map(str, sorted(lanes)))}"

        dir_parts = []
        for lane in sorted(lanes):
            d = directions.get(lane, "unknown")
            if d == "left":
                dir_parts.append(f"L{lane}: ← move right")
            elif d == "right":
                dir_parts.append(f"L{lane}: move left →")
            else:
                dir_parts.append(f"L{lane}: clear the lane")

        sub_text = f"🚨 {lane_str} — " + "  |  ".join(dir_parts)
        self._sub_lbl.configure(text=sub_text)
        self._title_lbl.configure(
            text=f"AMBULANCE IN {lane_str.upper()} — EMERGENCY PRIORITY"
        )

        if not self._flash_job:
            self._flash()

    def hide(self):
        """Hide the banner and stop flashing."""
        self._visible = False
        if self._flash_job:
            try:
                self.after_cancel(self._flash_job)
            except Exception:
                pass
            self._flash_job = None
        self.configure(fg_color=self._FLASH_COLORS[0])

    # ── Internal helpers ─────────────────────────────────────────────────

    def _flash(self):
        """Alternate between the two flash colours."""
        if not self._visible:
            self._flash_job = None
            return
        self._flash_idx = (self._flash_idx + 1) % 2
        color = self._FLASH_COLORS[self._flash_idx]
        self.configure(fg_color=color)
        self._flash_job = self.after(self._FLASH_INTERVAL_MS, self._flash)


class SignalTimerRow(ctk.CTkFrame):
    """A compact per-lane signal-state + countdown widget row.

    Displays lane number, state badge (GREEN/RED/YIELD/EMERGENCY), and
    recommended green time.
    """

    _STATE_COLORS = {
        "GREEN":     ("#1a7a3a", "#ffffff", "🟢 GREEN"),
        "RED":       ("#8b0000", "#ffffff", "🔴 RED"),
        "YIELD":     ("#b05000", "#ffffff", "🟡 YIELD"),
        "EMERGENCY": ("#7a0000", "#ffcccc", "🚨 CLEAR"),
    }

    def __init__(self, parent, lane: int, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.lane = lane

        ctk.CTkLabel(
            self,
            text=f"Lane {lane}",
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=theme.TEXT_2,
            width=52,
            anchor="w",
        ).pack(side="left")

        self._badge = ctk.CTkLabel(
            self,
            text="—",
            font=ctk.CTkFont("Segoe UI", 10, "bold"),
            width=80,
            corner_radius=4,
            fg_color=theme.BG_ELEVATED,
            text_color=theme.TEXT_3,
        )
        self._badge.pack(side="left", padx=(4, 0))

        self._timer_lbl = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont("Consolas", 10),
            text_color=theme.TEXT_3,
            anchor="e",
        )
        self._timer_lbl.pack(side="right", padx=(4, 0))

    def update_state(self, state: str, green_secs: int = 0, vehicle_count: int = 0):
        entry = self._STATE_COLORS.get(state, (theme.BG_ELEVATED, theme.TEXT_3, state))
        bg, fg, label = entry[0], entry[1], entry[2]
        self._badge.configure(text=label, fg_color=bg, text_color=fg)
        if state in ("GREEN",):
            self._timer_lbl.configure(text=f"{green_secs}s  |  {vehicle_count} veh")
        elif state == "EMERGENCY":
            self._timer_lbl.configure(text="🚨 PRIORITY")
        else:
            self._timer_lbl.configure(text=f"{vehicle_count} veh")
