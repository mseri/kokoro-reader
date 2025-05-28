# Kokoro Reader

A simple text-to-speech CLI tool using the Kokoro speech synthesis engine.

## Features

- High-quality text-to-speech synthesis
- Multiple voice options
- Automatic model download
- Interactive mode with command history
- Speed control
- Support for reading from files or standard input

## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation) for Python package management.
2. Clone or download this repository.

```bash
git clone https://github.com/yourusername/kokoro-reader.git
cd kokoro-reader
```

## Usage

### Basic Usage

```bash
uv run kokoro.py [options]
```

The required model files will be automatically downloaded to `~/.cache/kokoro-reader` on first run.

### Options

| Option | Description |
|--------|-------------|
| `-f, --file FILE` | Input text file (if not provided, reads from stdin) |
| `-v, --voice VOICE` | Voice to use (default: af_bella) |
| `-s, --speed SPEED` | Speech speed (default: 1.0) |
| `-l, --lang LANG` | Language (default: en-us) |
| `-i, --interactive` | Run in interactive mode |

### Examples

Read text from a file:
```bash
uv run kokoro.py -f mytext.txt
```

Use a specific voice:
```bash
uv run kokoro.py -v bf_emma -f mytext.txt
```

Adjust speech speed:
```bash
uv run kokoro.py -s 1.2 -f mytext.txt
```

Read from stdin:
```bash
echo "Hello, world!" | uv run kokoro.py
```

### Interactive Mode

Run in interactive mode:
```bash
uv run kokoro.py -i
```

In interactive mode, you can:
- Enter text directly (end with `/EOT`)
- Use arrow keys to navigate input history
- Edit input with left/right arrow keys
- Use commands to change settings

#### Interactive Commands

| Command | Description |
|---------|-------------|
| `TEXT` | Enter text directly (must end with `/EOT`) |
| `/f PATH` | Read text from file |
| `/v VOICE` | Change voice |
| `/v?` | Show available voices with grade C or better |
| `/l LANG` | Change language |
| `/s SPEED` | Change speed |
| `/q` | Quit |

## Available Voices

You can view all available high-quality voices by using the `/v?` command in interactive mode. The list includes American English, British English, and Italian voices.

For additional languages and voice options, see the official documentation: <https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>

### Voice Quality Grades

Voices are graded from A (best) to F (worst). Only voices with grade C or better are recommended.

### Voice Naming Convention

Voice names follow a specific naming convention:
- First letter: language (a=American, b=British, i=Italian)
- Second letter: gender (f=female, m=male)
- Followed by underscore and a name (e.g., af_bella = American Female Bella)

For complete voice listings and documentation, see <https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>

## Dependencies

- Python 3.12 or higher
- kokoro-onnx
- sounddevice
- requests
- tqdm
- prompt_toolkit

## Credits

- [Kokoro Speech Synthesis Engine](https://github.com/thewh1teagle/kokoro-onnx) - The underlying TTS library
- [Kokoro-82M Model](https://huggingface.co/hexgrad/Kokoro-82M) - Source of voice data and voice quality information
- This project was developed with assistance from Claude 3.7 Sonnet (via Copilot) using the Zed Editor's Agentic feature
