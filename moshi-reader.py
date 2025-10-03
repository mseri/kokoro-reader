#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx==0.2.12",
#     "numpy",
#     "sounddevice",
#     "requests",
#     "tqdm",
#     "prompt_toolkit",
#     "trafilatura",
#     "sentencepiece",
# ]
# ///
"""
Moshi Reader - High-quality text-to-speech CLI tool using Moshi MLX speech synthesis

Models are automatically downloaded from HuggingFace on first run.
By default, models are quantized to 8 bits for faster inference and lower memory usage.

Usage Examples:
  uv run moshi-reader.py -f mytext.txt             # Read from file (8-bit quantized)
  uv run moshi-reader.py -u https://example.com    # Read from URL
  echo "Hello world" | uv run moshi-reader.py      # Read from stdin
  uv run moshi-reader.py -i                        # Interactive mode
  uv run moshi-reader.py --quantize 0 -f text.txt  # No quantization (full precision)
  uv run moshi-reader.py --quantize 4 -f text.txt  # 4-bit quantization (more compressed)
"""

import sys
import asyncio
import argparse
import requests
import json
import queue
import time
from pathlib import Path
from tqdm import tqdm
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from dataclasses import dataclass
import typing as tp

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece
import sounddevice as sd
from trafilatura import fetch_url, extract

from moshi_mlx import models
from moshi_mlx.models.generate import LmGen
from moshi_mlx.client_utils import make_log
from moshi_mlx.modules.conditioner import (
    ConditionAttributes,
    ConditionTensor,
    dropout_all_conditions,
)
from moshi_mlx.utils.sampling import Sampler
from moshi_mlx.models.tts import (
    Entry,
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
    script_to_entries,
)
from moshi_mlx.utils.loaders import hf_get

MAX_LEN = 1024  # Maximum number of lines to read in interactive mode

class Voices:
    """Manage voice options for Moshi TTS"""

    def __init__(self):
        # Available voices from the moshi voice repo
        self.voices = {
            "expresso/ex03-ex01_happy_001_channel1_334s.wav": {"description": "Happy Expressive Female"},
            "expresso/ex03-ex01_sad_001_channel1_334s.wav": {"description": "Sad Expressive Female"},
            "expresso/ex03-ex01_neutral_001_channel1_334s.wav": {"description": "Neutral Expressive Female"},
            "expresso/ex03-ex01_angry_001_channel1_334s.wav": {"description": "Angry Expressive Female"},
            "default_male": {"description": "Default Male Voice"},
            "default_female": {"description": "Default Female Voice"},
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
            return f"{voice_name} ({info['description']})"
        return voice_name

    def display_available_voices(self):
        """Display list of available voices"""
        result = []
        result.append("Available voices:")
        result.append("Voice Name".ljust(50) + "Description")
        result.append("-" * 80)

        for voice_name, info in self.voices.items():
            result.append(f"{voice_name.ljust(48)} {info['description']}")

        return "\n".join(result)


def prepare_script(model: TTSModel, script: str, first_turn: bool) -> list[Entry]:
    multi_speaker = first_turn and model.multi_speaker
    return script_to_entries(
        model.tokenizer,
        model.machine.token_ids,
        model.mimi.frame_rate,
        [script],
        multi_speaker=multi_speaker,
        padding_between=1,
    )


def _make_null(
    all_attributes: tp.Sequence[ConditionAttributes],
) -> list[ConditionAttributes]:
    # When using CFG, returns the null conditions.
    return dropout_all_conditions(all_attributes)


@dataclass
class TTSGen:
    tts_model: TTSModel
    attributes: tp.Sequence[ConditionAttributes]
    on_frame: tp.Optional[tp.Callable[[mx.array], None]] = None

    def __post_init__(self):
        tts_model = self.tts_model
        attributes = self.attributes
        self.offset = 0
        self.state = self.tts_model.machine.new_state([])

        if tts_model.cfg_coef != 1.0:
            if tts_model.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG, but was trained with "
                    "CFG distillation. Pass instead `cfg_coef` to `make_condition_attributes`."
                )
            nulled = _make_null(attributes)
            attributes = list(attributes) + nulled

        assert tts_model.lm.condition_provider is not None
        self.ct = None
        self.cross_attention_src = None
        for _attr in attributes:
            for _key, _value in _attr.text.items():
                _ct = tts_model.lm.condition_provider.condition_tensor(_key, _value)
                if self.ct is None:
                    self.ct = _ct
                else:
                    self.ct = ConditionTensor(self.ct.tensor + _ct.tensor)
            for _key, _value in _attr.tensor.items():
                _conditioner = tts_model.lm.condition_provider.conditioners[_key]
                _ca_src = _conditioner.condition(_value)
                if self.cross_attention_src is None:
                    self.cross_attention_src = _ca_src
                else:
                    raise ValueError("multiple cross-attention conditioners")

        def _on_audio_hook(audio_tokens):
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[0]):
                delay = delays[q]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)

        self.lm_gen = LmGen(
            tts_model.lm,
            max_steps=tts_model.max_gen_length,
            text_sampler=Sampler(temp=tts_model.temp),
            audio_sampler=Sampler(temp=tts_model.temp),
            cfg_coef=tts_model.cfg_coef,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
        )

    def process_last(self):
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            self._step()
        additional_steps = (
            self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        )
        for _ in range(additional_steps):
            self._step()

    def process(self):
        while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
            self._step()

    def _step(self):
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = (
            mx.ones((1, missing), dtype=mx.int64)
            * self.tts_model.machine.token_ids.zero
        )
        self.lm_gen.step(
            input_tokens, ct=self.ct, cross_attention_src=self.cross_attention_src
        )
        frame = self.lm_gen.last_audio_tokens()
        self.offset += 1
        if frame is not None:
            if self.on_frame is not None:
                self.on_frame(frame)

    def append_entry(self, entry):
        self.state.entries.append(entry)


def log(level: str, msg: str):
    print(make_log(level, msg))


class MoshiTTS:
    """Moshi TTS wrapper class"""

    def __init__(self, hf_repo=DEFAULT_DSM_TTS_REPO, voice_repo=DEFAULT_DSM_TTS_VOICE_REPO, quantize=8):
        self.hf_repo = hf_repo
        self.voice_repo = voice_repo
        self.quantize = quantize
        self.tts_model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Moshi TTS model"""
        mx.random.seed(299792458)

        print("Initializing Moshi TTS...")

        raw_config = hf_get("config.json", self.hf_repo)
        with open(hf_get(raw_config), "r") as fobj:
            raw_config = json.load(fobj)

        mimi_weights = hf_get(raw_config["mimi_name"], self.hf_repo)
        moshi_name = raw_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_get(moshi_name, self.hf_repo)
        tokenizer = hf_get(raw_config["tokenizer_name"], self.hf_repo)
        lm_config = models.LmConfig.from_config_dict(raw_config)

        # There is a bug in moshi_mlx <= 0.3.0 handling of the ring kv cache.
        # The following line gets around it for now.
        lm_config.transformer.max_seq_len = lm_config.transformer.context
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)


        model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

        if self.quantize is not None and self.quantize > 0:

            nn.quantize(model.depformer, bits=self.quantize)
            for layer in model.transformer.layers:
                nn.quantize(layer.self_attn, bits=self.quantize)
                nn.quantize(layer.gating, bits=self.quantize)


        text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))


        generated_codebooks = lm_config.generated_codebooks
        audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
        audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

        cfg_coef_conditioning = None
        self.tts_model = TTSModel(
            model,
            audio_tokenizer,
            text_tokenizer,
            voice_repo=self.voice_repo,
            temp=0.6,
            cfg_coef=1,
            max_padding=8,
            initial_padding=2,
            final_padding=2,
            padding_bonus=0,
            raw_config=raw_config,
        )

        if self.tts_model.valid_cfg_conditionings:
            # Model was trained with CFG distillation.
            cfg_coef_conditioning = self.tts_model.cfg_coef
            self.tts_model.cfg_coef = 1.0

    def synthesize_and_play_streaming(self, text, voice="expresso/ex03-ex01_happy_001_channel1_334s.wav"):
        """Synthesize text and stream audio in real-time"""
        if not text or not text.strip():
            print("Error: Empty text provided")
            return False

        text = text.strip()
        print(f"Synthesizing: '{text[:50]}...'")

        try:
            if self.tts_model.multi_speaker:
                voices = [self.tts_model.get_voice_path(voice)]
            else:
                voices = []

            cfg_coef_conditioning = None
            if self.tts_model.valid_cfg_conditionings:
                cfg_coef_conditioning = self.tts_model.cfg_coef

            all_attributes = [
                self.tts_model.make_condition_attributes(voices, cfg_coef_conditioning)
            ]
        except Exception as e:
            print(f"Error setting up voice attributes: {e}")
            return False

        wav_frames = queue.Queue()
        frame_count = 0

        def _on_frame(frame):
            nonlocal frame_count
            if (frame == -1).any():
                return
            _pcm = self.tts_model.mimi.decode_step(frame[:, :, None])
            _pcm = np.array(mx.clip(_pcm[0, 0], -1, 1))
            wav_frames.put_nowait(_pcm)
            frame_count += 1

        gen = TTSGen(self.tts_model, all_attributes, on_frame=_on_frame)

        def audio_callback(outdata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            try:
                pcm_data = wav_frames.get(block=False)
                # Ensure pcm_data is the right type
                pcm_data = pcm_data.astype(np.float32)
                if len(pcm_data) <= frames:
                    outdata[:len(pcm_data), 0] = pcm_data
                    outdata[len(pcm_data):, 0] = 0
                else:
                    outdata[:, 0] = pcm_data[:frames]
                    # Put back remaining data
                    wav_frames.put_nowait(pcm_data[frames:])
            except queue.Empty:
                outdata[:, 0] = 0

        def run_synthesis():
            try:
                first_turn = True
                entries = prepare_script(self.tts_model, text, first_turn=first_turn)

                if not entries:
                    print("Error: No text entries created!")
                    return

                for entry in entries:
                    gen.append_entry(entry)
                    gen.process()

                gen.process_last()
            except Exception as e:
                print(f"Error during synthesis: {e}")

        try:
            # Check if audio device is available
            try:
                default_device = sd.default.device
                print(f"Using audio device: {default_device}")
            except Exception as device_error:
                print(f"Warning: Audio device check failed: {device_error}")

            # Start streaming audio output
            with sd.OutputStream(
                samplerate=self.tts_model.mimi.sample_rate,
                blocksize=1920,
                channels=1,
                callback=audio_callback,
                dtype=np.float32,
            ):
                run_synthesis()

                # Wait for audio to finish playing
                import time
                print("Playing audio...")
                while wav_frames.qsize() > 0:
                    time.sleep(0.1)
                time.sleep(1)  # Extra time for audio to finish

                print("Audio playback completed.")
                return True

        except Exception as e:
            print(f"Error during streaming: {e}")
            return False


def read_from_file(file_path, exit_on_error=True):
    """Read text from a file with robust error handling.

    Args:
        file_path: Path to the text file
        exit_on_error: If True, exit the program on error; otherwise return None

    Returns:
        String containing the file contents or None on error
    """
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
    """Extract readable text content from a URL.

    Uses trafilatura to extract the main content from a web page,
    filtering out navigation, ads, and other non-content elements.

    Args:
        url: The URL to fetch and extract content from
        exit_on_error: If True, exit the program on error; otherwise return None

    Returns:
        String containing the extracted text or None on error
    """
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


async def play_text(moshi_tts, text, voice):
    """Convert text to speech using Moshi and stream the resulting audio.

    Args:
        moshi_tts: Initialized Moshi TTS engine
        text: Text to convert to speech
        voice: Voice to use for synthesis
    """
    if not text.strip():
        print("Error: No text provided")
        return

    success = moshi_tts.synthesize_and_play_streaming(text, voice)
    if not success:
        print("Error: Failed to synthesize and play audio")


async def run_interactive_mode(moshi_tts, voice, voice_manager):
    """Run the application in interactive mode with persistent model and command history.

    Provides a command-line interface where users can enter text, change voices,
    and read from files or URLs without reloading the model.

    Args:
        moshi_tts: Initialized Moshi TTS engine
        voice: Initial voice to use
        voice_manager: Voice management object for validation and information
    """
    print("\n=== Moshi Interactive Mode ===")
    print("Available commands:")
    print("  TEXT          - Enter text directly (cannot start with '/', must end with /EOT)")
    print("  /f PATH       - Read text from file")
    print("  /u URL        - Read text from URL")
    print("  /v VOICE      - Change voice (current: {})".format(voice_manager.display_voice_info(voice)))
    print("  /v ?          - Show available voices")
    print("  /q            - Quit")
    print("\nUse up/down arrows to navigate history, and left/right arrows to edit input")

    current_voice = voice

    session = PromptSession(history=InMemoryHistory())

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
                        await play_text(moshi_tts, text, current_voice)
                    else:
                        print("Warning: Nothing to read.")

                elif cmd == '/u':
                    if not arg:
                        print("Error: Missing URL")
                        continue

                    text = extract_text_from_url(arg, exit_on_error=False)
                    if text:
                        await play_text(moshi_tts, text, current_voice)
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
                            print(f"Voice changed to: {voice_manager.display_voice_info(current_voice)}")
                        else:
                            print(f"Error: Voice '{arg}' not found. Use '/v ?' to see available voices.")

                else:
                    print(f"Unknown command: {cmd}")

            # Handle direct text input that doesn't start with /
            else:
                text_lines = [line]

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

                text = '\n'.join(text_lines)
                await play_text(moshi_tts, text, current_voice)

        except KeyboardInterrupt:
            print("\nInterrupted. Use /q to quit.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    voice_manager = Voices()

    parser = argparse.ArgumentParser(description='Text-to-speech using Moshi MLX')
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-f', '--file', help='Input text file')
    input_group.add_argument('-u', '--url', help='URL to extract text from')
    parser.epilog = 'Note: If neither -f/--file nor -u/--url is provided, text is read from standard input (stdin).'
    parser.add_argument('-v', '--voice',
                       default='expresso/ex03-ex01_happy_001_channel1_334s.wav',
                       help='Voice to use (use "/v ?" in interactive mode to see all available voices)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_DSM_TTS_REPO,
        help="HF repo in which to look for the pretrained models.",
    )
    parser.add_argument(
        "--voice-repo",
        default=DEFAULT_DSM_TTS_VOICE_REPO,
        help="HF repo in which to look for pre-computed voice embeddings.",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=8,
        choices=[0, 4, 8],
        help="The quantization to be applied: 0 for none, 4 for 4 bits, 8 for 8 bits (default: 8).",
    )

    args = parser.parse_args()

    if not voice_manager.validate_voice(args.voice):
        print(f"Error: Voice '{args.voice}' not found.")
        print("\nAvailable voices:")
        print(voice_manager.display_available_voices())
        sys.exit(1)

    quantize_value = args.quantize if args.quantize > 0 else None
    moshi_tts = MoshiTTS(args.hf_repo, args.voice_repo, quantize_value)

    if args.interactive:
        asyncio.run(run_interactive_mode(moshi_tts, args.voice, voice_manager))
    else:
        try:
            if args.file:
                text = read_from_file(args.file)
            elif args.url:
                text = extract_text_from_url(args.url)
            else:
                text = sys.stdin.read()

            if not text or not text.strip():
                print("Error: No text provided for processing", file=sys.stderr)
                sys.exit(1)

        except KeyboardInterrupt:
            print("\nInput interrupted. Exiting.", file=sys.stderr)
            sys.exit(1)

        asyncio.run(play_text(moshi_tts, text, args.voice))


if __name__ == "__main__":
    main()
