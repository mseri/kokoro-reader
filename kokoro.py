#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kokoro-onnx",
#   "sounddevice",
#   "requests",
#   "tqdm",
#   "prompt_toolkit",
# ]
# ///
"""
Usage:
1.
    Install uv from https://docs.astral.sh/uv/getting-started/installation
2.
    Copy this file to new folder
3.
    Run
    uv run kokoro.py

    (Required model files will be automatically downloaded to ~/.cache/kokoro-reader on first run)

For other languages read https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
"""

import asyncio
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

import sounddevice as sd

from kokoro_onnx import Kokoro

MAX_LEN = 1024  # Maximum number of lines to read in interactive mode

# Model URLs and cache location
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
CACHE_DIR = Path.home() / ".cache" / "kokoro-reader"
MODEL_PATH = CACHE_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = CACHE_DIR / "voices-v1.0.bin"

# Voice information (name, language, grade)
VOICE_INFO = {
    # American English
    "af_heart": {"lang": "en-us", "grade": "A"},
    "af_bella": {"lang": "en-us", "grade": "A-"},
    "af_nicole": {"lang": "en-us", "grade": "B-"},
    "af_aoede": {"lang": "en-us", "grade": "C+"},
    "af_kore": {"lang": "en-us", "grade": "C+"},
    "af_sarah": {"lang": "en-us", "grade": "C+"},
    "af_alloy": {"lang": "en-us", "grade": "C"},
    "af_nova": {"lang": "en-us", "grade": "C"},
    "af_sky": {"lang": "en-us", "grade": "C-"},
    "am_fenrir": {"lang": "en-us", "grade": "C+"},
    "am_michael": {"lang": "en-us", "grade": "C+"},
    "am_puck": {"lang": "en-us", "grade": "C+"},

    # British English
    "bf_emma": {"lang": "en-gb", "grade": "B-"},
    "bf_isabella": {"lang": "en-gb", "grade": "C"},
    "bm_fable": {"lang": "en-gb", "grade": "C"},
    "bm_george": {"lang": "en-gb", "grade": "C"},

    # Italian
    "if_sara": {"lang": "it", "grade": "C"},
    "im_nicola": {"lang": "it", "grade": "C"}
}

def ensure_model_files():
    """Download model files if they don't exist in the cache directory"""
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download model file if needed
    if not MODEL_PATH.exists():
        print(f"Downloading model file to {MODEL_PATH}...")
        download_file(MODEL_URL, MODEL_PATH)

    # Download voices file if needed
    if not VOICES_PATH.exists():
        print(f"Downloading voices file to {VOICES_PATH}...")
        download_file(VOICES_URL, VOICES_PATH)

    return str(MODEL_PATH), str(VOICES_PATH)

def download_file(url, destination_path):
    """Download a file from the given URL to the destination path"""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    if total_size == 0:
        raise ValueError(f"Failed to download file: Content-Length is 0 for {url}")

    with open(destination_path, 'wb') as file:
        desc = f"Downloading {destination_path.name}"
        with tqdm(total=total_size, unit='B', unit_scale=True,
                 desc=desc, ncols=80) as pbar:
            for data in response.iter_content(block_size):
                file.write(data)
                pbar.update(len(data))

async def main():
    parser = argparse.ArgumentParser(description='Text-to-speech using Kokoro')
    parser.add_argument('-f', '--file', help='Input text file (if not provided, reads from stdin)')
    parser.add_argument('-v', '--voice', default='af_bella', help='Voice to use (default: af_bella)')
    parser.add_argument('-s', '--speed', type=float, default=1.0, help='Speech speed (default: 1.0)')
    parser.add_argument('-l', '--lang', default='en-us', help='Language (default: en-us)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode')

    args = parser.parse_args()

    # Download models if needed and get their paths
    model_path, voices_path = ensure_model_files()

    # Initialize Kokoro
    print("Loading Kokoro model...")
    kokoro = Kokoro(model_path, voices_path)

    if args.interactive:
        await run_interactive_mode(kokoro, args.voice, args.speed, args.lang)
    else:
        # Read text from file or stdin
        try:
            if args.file:
                # Read from file
                try:
                    with open(args.file, 'r', encoding='utf-8') as f:
                        text = f.read()
                except FileNotFoundError:
                    print(f"Error: File '{args.file}' not found", file=sys.stderr)
                    sys.exit(1)
                except Exception as e:
                    print(f"Error reading file: {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                # Read from stdin
                text = sys.stdin.read()

            # Check if we got any text to process
            if not text.strip():
                print("Error: No text provided for processing", file=sys.stderr)
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nInput interrupted. Exiting.", file=sys.stderr)
            sys.exit(1)

        await play_text(kokoro, text, args.voice, args.speed, args.lang)


async def play_text(kokoro, text, voice, speed, lang):
    """Process text with Kokoro and play audio"""
    if not text.strip():
        print("Error: No text provided")
        return

    stream = kokoro.create_stream(
        text,
        voice=voice,
        speed=speed,
        lang=lang,
    )

    count = 0
    async for samples, sample_rate in stream:
        count += 1
        print(f"Playing audio stream ({count})...")
        sd.play(samples, sample_rate)
        sd.wait()

    print("Finished playing.")


async def run_interactive_mode(kokoro, voice, speed, lang):
    """Run in interactive mode where the model is loaded once and reused"""
    print("\n=== Kokoro Interactive Mode ===")
    print("Available commands:")
    print("  TEXT          - Enter text directly (cannot start with '/', must end with /EOT)")
    print("  /f PATH       - Read text from file")
    print("  /v VOICE      - Change voice (current: {})".format(voice))
    print("  /v?           - Show available voices with grade C or better")
    print("  /l LANG       - Change language (current: {})".format(lang))
    print("  /s SPEED      - Change speed (current: {})".format(speed))
    print("  /q            - Quit")
    print("\nUse up/down arrows to navigate history, and left/right arrows to edit input")

    current_voice = voice
    current_speed = speed
    current_lang = lang

    # Initialize prompt session with in-memory history
    session = PromptSession(history=InMemoryHistory())

    # Main interaction loop
    while True:
        try:
            line = await session.prompt_async("\n> ")

            # Handle empty input
            if not line:
                continue

            # Handle commands starting with /
            if line.startswith('/'):
                parts = line.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == '/q':
                    print("Exiting interactive mode.")
                    break

                elif cmd == '/f':
                    if not arg:
                        print("Error: Missing file path")
                        continue

                    try:
                        with open(arg, 'r', encoding='utf-8') as f:
                            text = f.read()
                        await play_text(kokoro, text, current_voice, current_speed, current_lang)
                    except FileNotFoundError:
                        print(f"Error: File '{arg}' not found")
                    except Exception as e:
                        print(f"Error reading file: {e}")

                elif cmd == '/v':
                    if arg == '?':
                        # Display list of voices with grade C or better
                        print("\nAvailable voices (grade C or better):")
                        print("Voice Name".ljust(15) + "Language".ljust(10) + "Grade")
                        print("-" * 30)

                        # Group by language for better organization
                        languages = {
                            "en-us": "American English",
                            "en-gb": "British English",
                            "it": "Italian"
                        }

                        for lang_code, lang_name in languages.items():
                            # Get voices for this language
                            lang_voices = {name: info for name, info in VOICE_INFO.items()
                                           if info["lang"] == lang_code}

                            if lang_voices:
                                print(f"\n{lang_name}:")
                                for voice_name, info in sorted(lang_voices.items()):
                                    print(f"  {voice_name.ljust(13)} {info['lang'].ljust(10)} {info['grade']}")
                    elif not arg:
                        print("Error: Missing voice parameter")
                        continue
                    else:
                        current_voice = arg
                        # Show language information if available
                        if current_voice in VOICE_INFO:
                            info = VOICE_INFO[current_voice]
                            print(f"Voice changed to: {current_voice} ({info['lang']}, grade: {info['grade']})")
                        else:
                            print(f"Voice changed to: {current_voice}")

                elif cmd == '/l':
                    if not arg:
                        print("Error: Missing language parameter")
                        continue
                    current_lang = arg
                    print(f"Language changed to: {current_lang}")

                elif cmd == '/s':
                    if not arg:
                        print("Error: Missing speed parameter")
                        continue
                    try:
                        current_speed = float(arg)
                        print(f"Speed changed to: {current_speed}")
                    except ValueError:
                        print("Error: Speed must be a number")

                else:
                    print(f"Unknown command: {cmd}")

            # Handle direct text input that doesn't start with /
            else:
                # Start collecting text
                text_lines = [line]

                # Continue reading until /EOT is found
                eot_found = False
                count = 1
                while not eot_found:
                  if count > MAX_LEN:
                    print("Error: Too many lines entered, stopping input collection.")
                    eot_found = True
                  else:
                    count += 1
                    line = await session.prompt_async("")
                    if line == '/EOT':
                        eot_found = True
                    else:
                        text_lines.append(line)

                # Process the collected text
                text = '\n'.join(text_lines)
                await play_text(kokoro, text, current_voice, current_speed, current_lang)

        except KeyboardInterrupt:
            print("\nInterrupted. Use /q to quit.")
        except Exception as e:
            print(f"Error: {e}")



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
