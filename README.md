# LiveKit Kokoro TTS Plugin

A high-performance text-to-speech plugin for LiveKit agents using Kokoro TTS with ultra-low latency streaming implementation for real-time voice synthesis.

## Features

- **Ultra-Low Latency**: ~80ms time-to-first-byte (TTFB) on RTX 4090
- **Multiple Voices**: Support for multiple voices and voice mixing
- **Streaming Support**: Real-time audio generation with chunked streaming
- **LiveKit Integration**: Seamless integration with LiveKit agents framework

## Requirements

- LiveKit Agents v1.0 or higher
- Kokoro FastAPI server instance
- NVIDIA GPU (recommended for optimal performance)
- Python 3.8+

## Performance

| Hardware | Latency | Quality | Use Case |
|----------|---------|---------|----------|
| **RTX 4090** | ~80ms TTFB | High | Real-time applications |

## Installation

1. Clone or download this plugin into your LiveKit-based agents project root directory
2. Set up the Kokoro FastAPI server for model inference
3. Install required dependencies:
   ```bash
   pip install openai httpx
   ```

## Server Setup

Use the Kokoro FastAPI server for optimized inference:

**Repository**: [remsky/Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)

This server provides OpenAI-compatible endpoints with optimized Kokoro TTS inference for ultra-low latency performance.

## Usage

Initialize your agent session with the KokoroTTS plugin:

```python
from kokoro_plugin import KokoroTTS

session = AgentSession(
    # ... other configuration
    tts=KokoroTTS(
        base_url="http://localhost:8000",
        api_key="NULL",
        voice="af_heart",
        speed=1.0
    )
)
```
