#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kokoro-onnx",
#   "sounddevice",
#   "requests",
#   "tqdm",
#   "prompt_toolkit",
#   "trafilatura",
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
    uv run --script kokoro-reader.py

    (Required model files will be automatically downloaded to ~/.cache/kokoro-reader on first run)

For other languages read https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

Input options (if not provided, reads from stdin):
- Read text from a file: -f FILE or --file FILE
- Read text from a URL: -u URL or --url URL
"""

import sys
import asyncio
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

import sounddevice as sd
from trafilatura import fetch_url, extract

from kokoro_onnx import Kokoro

MAX_LEN = 1024  # Maximum number of lines to read in interactive mode

# Model URLs and cache location
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
CACHE_DIR = Path.home() / ".cache" / "kokoro-reader"
MODEL_PATH = CACHE_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = CACHE_DIR / "voices-v1.0.bin"

class Voices:
    """Manage voice options and information for Kokoro TTS"""

    def __init__(self):
        # Voice information (name, language, grade)
        self.voices = {
            # American English
            "af_heart": {"lang": "en-us", "grade": "A"},
            "af_bella": {"lang": "en-us", "grade": "A-"},
            "af_nicole": {"lang": "en-us", "grade": "B-"},
            "af_alloy": {"lang": "en-us", "grade": "C"},
            "af_nova": {"lang": "en-us", "grade": "C"},
            "af_aoede": {"lang": "en-us", "grade": "C+"},
            "af_kore": {"lang": "en-us", "grade": "C+"},
            "af_sarah": {"lang": "en-us", "grade": "C+"},
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

        # Define language names for display
        self.languages = {
            "en-us": "American English",
            "en-gb": "British English",
            "it": "Italian"
        }

        # Define grade order for sorting (highest to lowest)
        self.grade_order = {
            'A+': 0, 'A': 1, 'A-': 2,
            'B+': 3, 'B': 4, 'B-': 5,
            'C+': 6, 'C': 7, 'C-': 8
        }

    def get_voice_info(self, voice_name):
        """Get information about a specific voice"""
        return self.voices.get(voice_name)

    def validate_voice(self, voice_name):
        """Check if a voice exists"""
        return voice_name in self.voices

    def display_voice_info(self, voice_name):
        """Return a formatted string with voice information"""
        if voice_name in self.voices:
            info = self.voices[voice_name]
            return f"{voice_name} ({info['lang']}, grade: {info['grade']})"
        return voice_name

    def display_available_voices(self):
        """Display list of available voices organized by language and gender"""
        result = []
        result.append("Available voices (grade C or better):")
        result.append("Voice Name".ljust(15) + "Language".ljust(10) + "Grade")
        result.append("-" * 30)

        # Function to sort voices by grade
        def sort_by_grade(voice_item):
            voice_name, voice_info = voice_item
            return self.grade_order[voice_info["grade"]]

        for lang_code, lang_name in self.languages.items():
            # Get voices for this language
            lang_voices = {name: info for name, info in self.voices.items()
                          if info["lang"] == lang_code}

            if lang_voices:
                result.append(f"\n{lang_name}:")

                # Separate female and male voices
                female_voices = {name: info for name, info in lang_voices.items()
                               if name.startswith(('af_', 'bf_', 'if_'))}
                male_voices = {name: info for name, info in lang_voices.items()
                             if name.startswith(('am_', 'bm_', 'im_'))}

                # Display female voices sorted by grade
                if female_voices:
                    result.append("  Female voices:")
                    for voice_name, info in sorted(female_voices.items(), key=sort_by_grade):
                        result.append(f"    {voice_name.ljust(13)} {info['lang'].ljust(10)} {info['grade']}")

                # Display male voices sorted by grade
                if male_voices:
                    result.append("  Male voices:")
                    for voice_name, info in sorted(male_voices.items(), key=sort_by_grade):
                        result.append(f"    {voice_name.ljust(13)} {info['lang'].ljust(10)} {info['grade']}")

        return "\n".join(result)

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

def read_from_file(file_path, exit_on_error=True):
    """Read text from a file and handle errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        error_msg = f"Error: File '{file_path}' not found"
        if exit_on_error:
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        else:
            print(error_msg)
            return None
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        if exit_on_error:
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        else:
            print(error_msg)
            return None

def extract_text_from_url(url, exit_on_error=True):
    """Extract text from a URL and handle errors"""
    try:
        print(f"Fetching content from URL: {url}")
        downloaded = fetch_url(url)
        if downloaded is None:
            error_msg = f"Error: Could not fetch URL '{url}'"
            if exit_on_error:
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            else:
                print(error_msg)
                return None

        text = extract(downloaded)
        if not text:
            error_msg = f"Error: Could not extract text from URL '{url}'"
            if exit_on_error:
                print(error_msg, file=sys.stderr)
                sys.exit(1)
            else:
                print(error_msg)
                return None

        print(f"Successfully extracted {len(text)} characters from URL")
        return text
    except Exception as e:
        error_msg = f"Error extracting text from URL: {e}"
        if exit_on_error:
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        else:
            print(error_msg)
            return None

def main():
    # Initialize voice manager
    voice_manager = Voices()
    
    parser = argparse.ArgumentParser(description='Text-to-speech using Kokoro')
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-f', '--file', help='Input text file')
    input_group.add_argument('-u', '--url', help='URL to extract text from')
    parser.epilog = 'Note: If neither -f/--file nor -u/--url is provided, text is read from standard input (stdin).'
    parser.add_argument('-v', '--voice', default='af_bella', help='Voice to use (default: af_bella)')
    parser.add_argument('-s', '--speed', type=float, default=0.8, help='Speech speed (default: 0.8)')
    parser.add_argument('-l', '--lang', default='en-us', help='Language (default: en-us)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode')

    args = parser.parse_args()
    
    # Validate the selected voice
    if not voice_manager.validate_voice(args.voice):
        print(f"Error: Voice '{args.voice}' not found.")
        print("\nAvailable voices:")
        print(voice_manager.display_available_voices())
        sys.exit(1)

    # Download models if needed and get their paths
    model_path, voices_path = ensure_model_files()

    # Initialize Kokoro
    print("Loading Kokoro model...")
    kokoro = Kokoro(model_path, voices_path)

    if args.interactive:
        asyncio.run(run_interactive_mode(kokoro, args.voice, args.speed, args.lang, voice_manager))
    else:
        # Read text from file, URL, or stdin
        try:
            if args.file:
                # Read from file
                text = read_from_file(args.file)
            elif args.url:
                # Extract text from URL
                text = extract_text_from_url(args.url)
            else:
                # Read from stdin
                text = sys.stdin.read()

            # Check if we got any text to process
            if not text or not text.strip():
                print("Error: No text provided for processing", file=sys.stderr)
                sys.exit(1)

        except KeyboardInterrupt:
            print("\nInput interrupted. Exiting.", file=sys.stderr)
            sys.exit(1)

        asyncio.run(play_text(kokoro, text, args.voice, args.speed, args.lang))


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


async def run_interactive_mode(kokoro, voice, speed, lang, voice_manager):
    """Run in interactive mode where the model is loaded once and reused"""
    print("\n=== Kokoro Interactive Mode ===")
    print("Available commands:")
    print("  TEXT          - Enter text directly (cannot start with '/', must end with /EOT)")
    print("  /f PATH       - Read text from file")
    print("  /u URL        - Read text from URL")
    print("  /v VOICE      - Change voice (current: {})".format(voice_manager.display_voice_info(voice)))
    print("  /v ?          - Show available voices with grade C or better")
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

                    text = read_from_file(arg, exit_on_error=False)
                    if text:
                        await play_text(kokoro, text, current_voice, current_speed, current_lang)
                    else:
                        print("Warning: Nothing to read.")

                elif cmd == '/u':
                    if not arg:
                        print("Error: Missing URL")
                        continue

                    text = extract_text_from_url(arg, exit_on_error=False)
                    if text:
                        await play_text(kokoro, text, current_voice, current_speed, current_lang)
                    else:
                        print("Warning: Nothing to read.")

                elif cmd == '/v':
                    if arg == '?':
                        print(voice_manager.display_available_voices())
                    elif not arg:
                        print("Error: Missing voice parameter")
                        continue
                    else:
                        if voice_manager.validate_voice(arg):
                            current_voice = arg
                            # Show language information if available
                            print(f"Voice changed to: {voice_manager.display_voice_info(current_voice)}")
                        else:
                            print(f"Error: Voice '{arg}' not found. Use '/v ?' to see available voices.")

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
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
