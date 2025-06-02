# Kokoro Reader

A simple text-to-speech CLI tool using the Kokoro speech synthesis engine.

## Features

- High-quality text-to-speech synthesis
- Multiple voice options with different languages and genders
- Automatic model download and caching
- Interactive mode with command history
- Speed control
- Support for reading from files, URLs, or standard input

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
| `-f, --file FILE` | Input text file |
| `-u, --url URL` | URL to extract text from |
| `-v, --voice VOICE` | Voice to use (default: af_bella) |
| `-s, --speed SPEED` | Speech speed (default: 0.8) |
| `-l, --lang LANG` | Language (default: en-us) |
| `-i, --interactive` | Run in interactive mode |

Note: If neither `-f/--file` nor `-u/--url` is provided, text is read from standard input (stdin).

### Examples

Read text from a file:
```bash
uv run kokoro.py -f mytext.txt
```

Read text from a URL (extracts main content from web pages):
```bash
uv run kokoro.py -u https://example.com/article
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
- Read text from files or URLs without exiting the application
- Use arrow keys to navigate input history
- Edit input with left/right arrow keys
- Use commands to change voice, language and speed settings

#### Interactive Commands

| Command | Description |
|---------|-------------|
| `TEXT` | Enter text directly (must end with `/EOT`) |
| `/f PATH` | Read text from file |
| `/u URL` | Read text from URL |
| `/v VOICE` | Change voice |
| `/v?` | Show available voices with grade C or better |
| `/l LANG` | Change language |
| `/s SPEED` | Change speed |
| `/q` | Quit |

## Available Voices

You can view all available high-quality voices by using the `/v?` command in interactive mode or checking the table below. The list includes American English, British English, and Italian voices, organized by gender and sorted by quality.

For additional languages and voice options, see the official documentation: <https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>

### Voice Quality Grades

Voices are graded from A (best) to F (worst). Only voices with grade C or better are recommended.

### Best Voices by Language

#### American English (en-us)
| Voice Name | Gender | Grade | Description |
|------------|--------|-------|-------------|
| af_heart   | Female | A     | Best overall voice quality |
| af_bella   | Female | A-    | Default voice, excellent quality |
| af_nicole  | Female | B-    | Good quality |
| am_fenrir  | Male   | C+    | Best male voice for American English |

#### British English (en-gb)
| Voice Name  | Gender | Grade |
|-------------|--------|-------|
| bf_emma     | Female | B-    |
| bf_isabella | Female | C     |
| bm_fable    | Male   | C     |

#### Italian (it)
| Voice Name | Gender | Grade |
|------------|--------|-------|
| if_sara    | Female | C     |
| im_nicola  | Male   | C     |

### Voice Naming Convention

Voice names follow a specific naming convention:
- First letter: language (a=American, b=British, i=Italian)
- Second letter: gender (f=female, m=male)
- Followed by underscore and a name (e.g., af_bella = American Female Bella)

For complete voice listings and documentation, see <https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>

## Dependencies

- Python 3.12 or higher
- kokoro-onnx - Core speech synthesis library
- sounddevice - Audio output
- requests - Network requests for downloading models
- tqdm - Progress bars for downloads
- prompt_toolkit - Interactive shell interface
- trafilatura - Web content extraction for URL reading

## Credits

- [Kokoro Speech Synthesis Engine](https://github.com/thewh1teagle/kokoro-onnx) - The underlying TTS library
- [Kokoro-82M Model](https://huggingface.co/hexgrad/Kokoro-82M) - Source of voice data and voice quality information
- [Trafilatura](https://github.com/adbar/trafilatura) - Used for extracting readable content from web pages
- This project was developed with assistance from Claude 3.7 Sonnet (via Copilot) using the Zed Editor's Agentic feature
