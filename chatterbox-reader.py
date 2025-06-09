#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "chatterbox-streaming",
#   "sounddevice",
#   "torch",
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
    uv run kokoro.py

    (Required model files will be automatically downloaded to ~/.cache/kokoro-reader on first run)

For other languages read https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

Input options (if not provided, reads from stdin):
- Read text from a file: -f FILE or --file FILE
- Read text from a URL: -u URL or --url URL
"""

import sys
import asyncio
import argparse
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

import sounddevice as sd
from trafilatura import fetch_url, extract

import torch
import numpy as np
from chatterbox.tts import ChatterboxTTS

MAX_LEN = 1024  # Maximum number of lines to read in interactive mode

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
    parser = argparse.ArgumentParser(description='Text-to-speech using Kokoro')
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-f', '--file', help='Input text file')
    input_group.add_argument('-u', '--url', help='URL to extract text from')
    parser.epilog = 'Note: If neither -f/--file nor -u/--url is provided, text is read from standard input (stdin).'
    parser.add_argument('-s', '--speed', type=float, default=0.5, help='Speed of speech (default: 0.5)')
    parser.add_argument('-e', '--exaggeration', type=float, default=0.5, help='Emotional exaggeration (default: 0.5)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode')

    args = parser.parse_args()

    # Detect device (Mac with M1/M2/M3/M4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    map_location = torch.device(device)

    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)

    torch.load = patched_torch_load

    model = ChatterboxTTS.from_pretrained(device=device)

    if args.interactive:
        asyncio.run(run_interactive_mode(model, args.exaggeration, args.speed))
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

        asyncio.run(play_text(model, text, args.exaggeration, args.speed))

async def play_text(model, text, exaggeration, cfg_weight):
    """Process text with Kokoro and play audio"""
    if not text.strip():
        print("Error: No text provided")
        return

    # Basic streaming
    stream = model.generate_stream(text, exaggeration=exaggeration, cfg_weight=cfg_weight)

    count = 0
    for sample, metrics in stream:
        count += 1
        print(f"Playing audio stream ({count})...")
        sample = sample.numpy().astype(np.float32).T
        sd.play(sample, model.sr)
        sd.wait()

    print("Finished playing.")


async def run_interactive_mode(model, exaggeration, cfg_weight):
    """Run in interactive mode where the model is loaded once and reused"""
    print("\n=== Kokoro Interactive Mode ===")
    print("Available commands:")
    print("  TEXT            - Enter text directly (cannot start with '/', must end with /EOT)")
    print("  /f PATH         - Read text from file")
    print("  /u URL          - Read text from URL")
    print("  /e EXAGGERATION - Change exaggeration (current: {})".format(exaggeration))
    print("  /s SPEED        - Change speed (current: {})".format(cfg_weight))
    print("  /q              - Quit")
    print("\nUse up/down arrows to navigate history, and left/right arrows to edit input")

    current_exaggeration = exaggeration
    current_speed = cfg_weight

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
                        await play_text(model, text, current_exaggeration, current_speed)
                    else:
                        print("Warning: Nothing to read.")

                elif cmd == '/u':
                    if not arg:
                        print("Error: Missing URL")
                        continue

                    text = extract_text_from_url(arg, exit_on_error=False)
                    if text:
                        await play_text(model, text, current_exaggeration, current_speed)
                    else:
                        print("Warning: Nothing to read.")

                elif cmd == '/e':
                    if not arg:
                        print("Error: Missing exaggeration parameter")
                        continue
                    current_exaggeration = arg
                    print(f"Exaggeration changed to: {current_exaggeration}")

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
                await play_text(model, text, current_exaggeration, current_speed)

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
