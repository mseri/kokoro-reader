#!/usr/bin/env python3
import asyncio
import sys
import argparse
import os

import sounddevice as sd

from kokoro_onnx import Kokoro


async def main():
    parser = argparse.ArgumentParser(description='Text-to-speech using Kokoro')
    parser.add_argument('-f', '--file', help='Input text file (if not provided, reads from stdin)')
    parser.add_argument('-v', '--voice', default='af_bella', help='Voice to use (default: af_bella)')
    parser.add_argument('-s', '--speed', type=float, default=1.0, help='Speech speed (default: 1.0)')
    parser.add_argument('-l', '--lang', default='en-us', help='Language (default: en-us)')
    
    args = parser.parse_args()
    
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
    
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

    stream = kokoro.create_stream(
        text,
        voice=args.voice,
        speed=args.speed,
        lang=args.lang,
    )

    count = 0
    async for samples, sample_rate in stream:
        count += 1
        print(f"Playing audio stream ({count})...")
        sd.play(samples, sample_rate)
        sd.wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
