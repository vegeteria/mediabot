# Telegram Media Download Bot - Walkthrough

## What Was Built

A fully-featured Telegram bot for managing media downloads on a Linux server.

### Files Created

| File | Purpose |
|------|---------|
| [media_bot.py] | Main bot script (450+ lines) |
| [requirements.txt] | Python dependencies |
| [.env.example] | Environment template |

---

## Features Implemented

| Feature | Description |
|---------|-------------|
| **Authorization** | Owner + authorized users system with `/authorize` and `/deauthorize` |
| **Live Progress** | `[■■■■□□□□□□] 40%` with speed and ETA |
| **Movie Downloads** | Direct file download to `~/server/movies/` |
| **Series Downloads** | Archive extraction with Season folder organization |
| **Smart Detection** | Parses `S01E01`, `Season 1`, `1x01` patterns |

---

## Setup Instructions

### 1. Install System Dependencies
```bash
sudo apt install p7zip-full python3-venv
```

### 2. Create Virtual Environment
```bash
cd path/to/mediabot
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
nano .env
```

Set your values:
```
BOT_TOKEN=123456:ABC-DEF...  # From @BotFather
OWNER_ID=123456789           # Your Telegram user ID
```

> [!TIP]
> Get your user ID by sending a message to [@userinfobot](https://t.me/userinfobot)

### 5. Run the Bot
```bash
# Make sure venv is activated
source venv/bin/activate
python3 media_bot.py
```

---

## Commands Reference

| Command | Access | Description |
|---------|--------|-------------|
| `/start` | Everyone | Show status and help |
| `/movie <url>` | Authorized | Download file to movies folder |
| `/series <name> <url>` | Authorized | Download, extract, and organize series |
| `/authorize <id>` | Authorized | Grant access to a user |
| `/deauthorize <id>` | Owner only | Revoke user access |

---

## Usage Examples

**Download a movie:**
```
/movie https://example.com/movie.mkv
```

**Download a series:**
```
/series Breaking.Bad https://example.com/bb.zip
```

**Authorize a friend:**
```
/authorize 987654321
```

---

## Validation Results

- ✅ Python syntax check passed
- ✅ All dependencies specified
- ✅ Authorization system with persistence
- ✅ Progress tracking with throttled updates
- ✅ Season detection with multiple patterns
