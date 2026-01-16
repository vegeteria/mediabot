#!/usr/bin/env python3
"""
Telegram Media Download Bot
Downloads movies and series with progress tracking and smart folder organization.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Set
from urllib.parse import unquote, urlparse

import aiohttp
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Load environment variables
load_dotenv()

# Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
BASE_MOVIES = Path.home() / "server" / "movies"
BASE_SERIES = Path.home() / "server" / "series"
AUTH_FILE = Path(__file__).parent / "authorized_users.json"

# Video file extensions
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}

# Season patterns for detection
SEASON_PATTERNS = [
    re.compile(r"[Ss](\d{1,2})[Ee]\d{1,3}"),           # S01E01
    re.compile(r"[Ss]eason[\s._-]*(\d{1,2})", re.I),   # Season 1, Season.1
    re.compile(r"[\s._-](\d{1,2})x\d{1,3}[\s._-]"),    # 1x01
]

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class AuthManager:
    """Manages authorized users with persistence."""
    
    def __init__(self, filepath: Path, owner_id: int):
        self.filepath = filepath
        self.owner_id = owner_id
        self.authorized_users: Set[int] = set()
        self._load()
    
    def _load(self):
        """Load authorized users from file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    self.authorized_users = set(data.get("users", []))
            except (json.JSONDecodeError, IOError):
                self.authorized_users = set()
    
    def _save(self):
        """Save authorized users to file."""
        with open(self.filepath, "w") as f:
            json.dump({"users": list(self.authorized_users)}, f, indent=2)
    
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized (owner always is)."""
        return user_id == self.owner_id or user_id in self.authorized_users
    
    def authorize(self, user_id: int) -> bool:
        """Add user to authorized list. Returns True if newly added."""
        if user_id not in self.authorized_users:
            self.authorized_users.add(user_id)
            self._save()
            return True
        return False
    
    def deauthorize(self, user_id: int) -> bool:
        """Remove user from authorized list. Returns True if removed."""
        if user_id in self.authorized_users:
            self.authorized_users.discard(user_id)
            self._save()
            return True
        return False


class ProgressTracker:
    """Tracks download progress and updates Telegram message."""
    
    def __init__(self, message, total_size: int):
        self.message = message
        self.total_size = total_size
        self.downloaded = 0
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 2  # seconds
    
    def _format_size(self, size: int) -> str:
        """Format bytes to human readable."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to human readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"
    
    async def update(self, chunk_size: int):
        """Update progress with new chunk."""
        self.downloaded += chunk_size
        current_time = time.time()
        
        # Throttle updates
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # Calculate progress
        if self.total_size > 0:
            percent = min(100, (self.downloaded / self.total_size) * 100)
            filled = int(percent / 10)
            bar = "■" * filled + "□" * (10 - filled)
        else:
            percent = 0
            bar = "□" * 10
        
        # Calculate speed and ETA
        elapsed = current_time - self.start_time
        if elapsed > 0:
            speed = self.downloaded / elapsed
            speed_str = f"{self._format_size(speed)}/s"
            
            if self.total_size > 0 and speed > 0:
                remaining = (self.total_size - self.downloaded) / speed
                eta_str = f"ETA: {self._format_time(remaining)}"
            else:
                eta_str = ""
        else:
            speed_str = "calculating..."
            eta_str = ""
        
        # Build progress message
        progress_msg = (
            f"📥 Downloading...\n\n"
            f"[{bar}] {percent:.1f}%\n"
            f"📊 {self._format_size(self.downloaded)}"
        )
        if self.total_size > 0:
            progress_msg += f" / {self._format_size(self.total_size)}"
        progress_msg += f"\n⚡ {speed_str}"
        if eta_str:
            progress_msg += f"\n⏱ {eta_str}"
        
        try:
            await self.message.edit_text(progress_msg)
        except Exception:
            pass  # Ignore rate limit or same content errors


class AsyncDownloader:
    """Handles async file downloads with progress tracking."""
    
    @staticmethod
    def extract_filename(url: str, headers: dict) -> str:
        """Extract filename from URL or Content-Disposition header."""
        # Try Content-Disposition header first
        cd = headers.get("Content-Disposition", "")
        if "filename=" in cd:
            match = re.search(r'filename[*]?=["\']?([^"\';]+)', cd)
            if match:
                return unquote(match.group(1))
        
        # Fall back to URL path
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = os.path.basename(path)
        
        return filename if filename else "download"
    
    @staticmethod
    async def download(url: str, dest_dir: Path, progress_tracker: Optional[ProgressTracker] = None) -> Path:
        """Download file from URL to destination directory."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        timeout = aiohttp.ClientTimeout(total=None, connect=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, allow_redirects=True) as response:
                response.raise_for_status()
                
                # Get filename and total size
                filename = AsyncDownloader.extract_filename(url, dict(response.headers))
                total_size = int(response.headers.get("Content-Length", 0))
                
                dest_path = dest_dir / filename
                
                # Update tracker with size
                if progress_tracker:
                    progress_tracker.total_size = total_size
                
                # Stream download
                with open(dest_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                        if progress_tracker:
                            await progress_tracker.update(len(chunk))
                
                return dest_path


class SeasonOrganizer:
    """Organizes video files into season folders."""
    
    @staticmethod
    def detect_season(filename: str) -> Optional[int]:
        """Detect season number from filename."""
        for pattern in SEASON_PATTERNS:
            match = pattern.search(filename)
            if match:
                return int(match.group(1))
        return None
    
    @staticmethod
    def organize(series_dir: Path) -> dict:
        """Organize extracted videos into Season X folders."""
        stats = {"moved": 0, "unknown": 0}
        
        # Find all video files recursively
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(series_dir.rglob(f"*{ext}"))
        
        for video_path in video_files:
            season = SeasonOrganizer.detect_season(video_path.name)
            
            if season is not None:
                season_dir = series_dir / f"Season {season}"
                season_dir.mkdir(exist_ok=True)
                
                dest = season_dir / video_path.name
                if dest != video_path:
                    shutil.move(str(video_path), str(dest))
                    stats["moved"] += 1
            else:
                # Move unknown season files to root
                dest = series_dir / video_path.name
                if dest != video_path:
                    shutil.move(str(video_path), str(dest))
                    stats["unknown"] += 1
        
        # Cleanup empty directories
        for dirpath, dirnames, filenames in os.walk(series_dir, topdown=False):
            dir_path = Path(dirpath)
            if dir_path != series_dir and not any(dir_path.iterdir()):
                dir_path.rmdir()
        
        return stats


def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


# Initialize auth manager
auth_manager = AuthManager(AUTH_FILE, OWNER_ID)


def require_auth(func):
    """Decorator to require authorization for commands."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not auth_manager.is_authorized(user_id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            return
        return await func(update, context)
    return wrapper


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user_id = update.effective_user.id
    is_auth = auth_manager.is_authorized(user_id)
    
    msg = (
        "🎬 **Media Download Bot**\n\n"
        f"Your User ID: `{user_id}`\n"
        f"Status: {'✅ Authorized' if is_auth else '⛔ Not Authorized'}\n\n"
    )
    
    if is_auth:
        msg += (
            "**Commands:**\n"
            "• `/movie <url>` - Download movie\n"
            "• `/series <name> <url>` - Download & extract series\n"
            "• `/authorize <user_id>` - Authorize a user\n"
            "• `/deauthorize <user_id>` - Remove user access"
        )
    else:
        msg += "Contact the bot owner to get authorized."
    
    await update.message.reply_text(msg, parse_mode="Markdown")


@require_auth
async def authorize_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /authorize command."""
    if not context.args:
        await update.message.reply_text("Usage: `/authorize <user_id>`", parse_mode="Markdown")
        return
    
    try:
        target_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid user ID. Must be a number.")
        return
    
    if auth_manager.authorize(target_id):
        await update.message.reply_text(f"✅ User `{target_id}` has been authorized.", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"ℹ️ User `{target_id}` is already authorized.", parse_mode="Markdown")


@require_auth
async def deauthorize_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /deauthorize command."""
    user_id = update.effective_user.id
    
    # Only owner can deauthorize
    if user_id != OWNER_ID:
        await update.message.reply_text("⛔ Only the owner can deauthorize users.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: `/deauthorize <user_id>`", parse_mode="Markdown")
        return
    
    try:
        target_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid user ID. Must be a number.")
        return
    
    if target_id == OWNER_ID:
        await update.message.reply_text("❌ Cannot deauthorize the owner.")
        return
    
    if auth_manager.deauthorize(target_id):
        await update.message.reply_text(f"✅ User `{target_id}` has been deauthorized.", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"ℹ️ User `{target_id}` was not authorized.", parse_mode="Markdown")


@require_auth
async def download_movie(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /movie command."""
    if not context.args:
        await update.message.reply_text("Usage: `/movie <url>`", parse_mode="Markdown")
        return
    
    url = context.args[0]
    
    if not validate_url(url):
        await update.message.reply_text("❌ Invalid URL format.")
        return
    
    status_msg = await update.message.reply_text("📥 Starting download...")
    
    try:
        tracker = ProgressTracker(status_msg, 0)
        filepath = await AsyncDownloader.download(url, BASE_MOVIES, tracker)
        
        await status_msg.edit_text(
            f"✅ Download complete!\n\n"
            f"📁 **File:** `{filepath.name}`\n"
            f"📍 **Location:** `{filepath.parent}`",
            parse_mode="Markdown"
        )
    except aiohttp.ClientError as e:
        await status_msg.edit_text(f"❌ Download failed: {str(e)}")
    except Exception as e:
        logger.exception("Movie download error")
        await status_msg.edit_text(f"❌ Error: {str(e)}")


@require_auth
async def download_series(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /series command."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: `/series <name> <url>`\n\n"
            "Example: `/series Breaking.Bad https://example.com/bb.zip`",
            parse_mode="Markdown"
        )
        return
    
    series_name = context.args[0]
    url = context.args[1]
    
    # Sanitize series name
    series_name = re.sub(r'[<>:"/\\|?*]', '_', series_name)
    
    if not validate_url(url):
        await update.message.reply_text("❌ Invalid URL format.")
        return
    
    series_dir = BASE_SERIES / series_name
    status_msg = await update.message.reply_text(f"📁 Creating folder: `{series_name}`...", parse_mode="Markdown")
    
    try:
        series_dir.mkdir(parents=True, exist_ok=True)
        
        # Download archive
        await status_msg.edit_text("📥 Downloading archive...")
        tracker = ProgressTracker(status_msg, 0)
        archive_path = await AsyncDownloader.download(url, series_dir, tracker)
        
        # Extract archive
        await status_msg.edit_text("📦 Extracting archive...")
        
        extract_dir = series_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        # Use 7z for extraction (handles zip, rar, 7z, etc.)
        result = subprocess.run(
            ["7z", "x", str(archive_path), f"-o{extract_dir}", "-y"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Extraction failed: {result.stderr}")
        
        # Organize into season folders
        await status_msg.edit_text("🗂 Organizing files...")
        
        # Move extracted files to series root for organization
        for item in extract_dir.iterdir():
            dest = series_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        
        # Remove empty extracted dir
        if extract_dir.exists() and not any(extract_dir.iterdir()):
            extract_dir.rmdir()
        
        # Organize videos into Season X folders
        stats = SeasonOrganizer.organize(series_dir)
        
        # Cleanup archive
        archive_path.unlink(missing_ok=True)
        
        # Build result message
        result_msg = (
            f"✅ Series ready!\n\n"
            f"📁 **Series:** `{series_name}`\n"
            f"📍 **Location:** `{series_dir}`\n\n"
            f"📊 **Organized:**\n"
            f"• {stats['moved']} files moved to Season folders\n"
        )
        if stats['unknown'] > 0:
            result_msg += f"• {stats['unknown']} files with unknown season"
        
        await status_msg.edit_text(result_msg, parse_mode="Markdown")
        
    except subprocess.CalledProcessError as e:
        await status_msg.edit_text(f"❌ Extraction failed: {e}")
    except aiohttp.ClientError as e:
        await status_msg.edit_text(f"❌ Download failed: {str(e)}")
    except Exception as e:
        logger.exception("Series download error")
        await status_msg.edit_text(f"❌ Error: {str(e)}")


def main():
    """Start the bot."""
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not set in environment!")
        return
    
    if OWNER_ID == 0:
        logger.error("OWNER_ID not set in environment!")
        return
    
    # Ensure base directories exist
    BASE_MOVIES.mkdir(parents=True, exist_ok=True)
    BASE_SERIES.mkdir(parents=True, exist_ok=True)
    
    # Build application
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("authorize", authorize_user))
    app.add_handler(CommandHandler("deauthorize", deauthorize_user))
    app.add_handler(CommandHandler("movie", download_movie))
    app.add_handler(CommandHandler("series", download_series))
    
    # Start polling
    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
