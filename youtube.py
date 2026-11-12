import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import yt_dlp
from utils import logger


class DownloadCancelledException(Exception):
    """Exception raised when user cancels a YouTube download."""
    pass


def _cookie_opts() -> dict:
    """
    Try to return a yt-dlp option dict that supplies cookies from the user's
    default browser so YouTube does not return HTTP 403.

    Tries Chrome → Edge → Firefox → Brave → Opera in that order.
    Falls back to an empty dict (no cookies) if no browser is available so
    non-restricted videos still work.
    """
    for browser in ("chrome", "edge", "firefox", "brave", "opera"):
        try:
            import yt_dlp
            # Dry-run to test if browser cookies are extractable
            probe_opts = {"cookiesfrombrowser": (browser,), "quiet": True, "no_warnings": True}
            # Cookie extraction can print Chromium DPAPI errors directly to
            # stderr even when yt-dlp is quiet. It is an optional probe, so
            # keep those expected failures out of the user's console.
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with yt_dlp.YoutubeDL(probe_opts) as ydl:
                    ydl.cookiejar  # triggers cookie extraction; raises if unavailable
            logger.info("YouTube: using cookies from '%s' browser.", browser)
            return {"cookiesfrombrowser": (browser,)}
        except Exception:
            continue
    logger.warning("YouTube: no browser cookies found; downloads may fail with HTTP 403.")
    return {}


class YouTubeDownloader:
    """Manages downloading YouTube videos with progress callbacks and cancel support."""
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        """Sets the cancellation flag to interrupt downloads."""
        self._cancelled = True
        logger.info("Cancellation requested for YouTube download.")

    def download(self, url, output_dir, progress_callback=None):
        """
        Downloads a YouTube video to the output directory.
        
        Args:
            url (str): YouTube video URL.
            output_dir (str): Directory where downloaded file is saved.
            progress_callback (callable): Function accepting (percent_float, speed_str, eta_str, status_str).
            
        Returns:
            str: Path to the downloaded video file.
        """
        self._cancelled = False
        os.makedirs(output_dir, exist_ok=True)
        
        # Titles may contain characters that Windows rejects. The video ID is
        # stable and unique, so use it as the portable filename.
        outtmpl = os.path.join(output_dir, '%(id)s.%(ext)s')
        ffmpeg_available = shutil.which("ffmpeg") is not None

        def ytdl_hook(d):
            if self._cancelled:
                raise DownloadCancelledException("Download cancelled by user.")
            
            if d['status'] == 'downloading':
                total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                downloaded_bytes = d.get('downloaded_bytes', 0)
                
                percent = 0.0
                if total_bytes > 0:
                    percent = (downloaded_bytes / total_bytes) * 100
                
                # Format speed string
                speed = d.get('speed', 0)
                speed_str = "0 B/s"
                if speed:
                    if speed > 1024 * 1024:
                        speed_str = f"{speed / (1024 * 1024):.1f} MB/s"
                    elif speed > 1024:
                        speed_str = f"{speed / 1024:.1f} KB/s"
                    else:
                        speed_str = f"{speed:.0f} B/s"
                
                # Format ETA
                eta = d.get('eta', 0)
                eta_str = f"{eta}s" if eta else "Unknown"
                
                if progress_callback:
                    progress_callback(percent, speed_str, eta_str, "Downloading")
            
            elif d['status'] == 'finished':
                if progress_callback:
                    progress_callback(100.0, "0 B/s", "0s", "Download finished. Merging formats...")

        # Use a single progressive stream when FFmpeg is unavailable. With
        # FFmpeg, separate streams give better quality and are merged to MP4.
        if ffmpeg_available:
            format_selector = "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
        else:
            format_selector = "best[ext=mp4][height<=720]/best[ext=mp4]/best[height<=720]/best"

        # Build base options — merge in browser cookies to beat 403 blocks
        cookie_opts = _cookie_opts()
        ydl_opts = {
            'format': format_selector,
            'outtmpl': outtmpl,
            'windowsfilenames': True,
            'progress_hooks': [ytdl_hook],
            'noprogress': True,
            'quiet': True,
            'no_warnings': True,
            # Retries help with transient server hiccups
            'retries': 3,
            'fragment_retries': 3,
            # Some YouTube throttling is bypassed with this
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            **cookie_opts,
        }
        if ffmpeg_available:
            ydl_opts['merge_output_format'] = 'mp4'

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = info.get('filepath') or ydl.prepare_filename(info)

                # Post-processing can change the extension; newer yt-dlp
                # versions also expose the final path on requested downloads.
                candidates = [filename]
                candidates.extend(
                    item.get('filepath', '')
                    for item in info.get('requested_downloads', [])
                    if isinstance(item, dict)
                )
                base_filepath, _ = os.path.splitext(filename)
                candidates.extend(base_filepath + ext for ext in ('.mp4', '.mkv', '.webm', '.avi'))
                for candidate in candidates:
                    if candidate and os.path.exists(candidate):
                        candidate = os.path.abspath(candidate)
                        logger.info("YouTube video ready: %s", candidate)
                        return candidate

                raise FileNotFoundError(
                    "YouTube video download finished but output file was not found."
                )
                    
        except DownloadCancelledException:
            raise
        except Exception as e:
            logger.error(f"yt-dlp download failed for URL {url}: {e}")
            raise
