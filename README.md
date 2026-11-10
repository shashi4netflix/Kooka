# Kooka

A desktop voice assistant for Windows with a chat-style window. Say "Kooka" (or type) to ask a question; Kooka answers out loud, can search the web, show pictures, and — in the video version — play YouTube videos inside the app.

## Two versions

| File | What it includes |
|---|---|
| `Agent Box 2.py` | Everything, including in-app videos |
| `Agent Box 2 - No Video.py` | Everything except videos (fewer things to install) |

## Setup

1. Install Python 3, then the packages:
   ```
   python -m pip install -r requirements.txt
   ```
2. Download two [Vosk](https://alphacephei.com/vosk/models) speech models and place them next to the program:
   - a language model in a folder named `model`
   - the speaker model in a folder named `model-spk`
3. Create a file named `groq_api_key.txt` next to the program containing your [Groq](https://console.groq.com/) API key (or set the `GROQ_API_KEY` environment variable). This file is listed in `.gitignore` so it is never uploaded.
4. For the video version only: install the 64-bit [VLC media player](https://www.videolan.org/) and Deno (`winget install DenoLand.Deno`).

## Run

```
python "Agent Box 2.py"
```

Then click **Set up my voice** in the sidebar and repeat the 10 phrases — Kooka only responds to spoken commands from the enrolled voice.

## Private files (not in this repository)

Saved chats (`datalinks.json`), your voice profile (`voice_blueprint.json`), settings, your API key, and the speech models stay on your computer only.
