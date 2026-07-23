# Azure Speech Benchmarking

A Python tool for benchmarking **Azure Speech-to-Text (STT)** streaming performance under both sequential and concurrent load. It measures key latency metrics — Time to First Token (TTFT), Time to Final Result (TTFR), and STT latency — while streaming audio to the Azure Speech service in real-time-simulated chunks.

## Features

- **Real-time streaming** using the Azure Speech SDK `PushAudioInputStream` (100 ms chunks).
- **Sequential latency testing** to establish baseline TTFT / TTFR / latency.
- **Concurrent load testing** across increasing concurrency levels (10 → 500 simultaneous streams).
- **Detailed metrics**: time to first token, time to final result, per-request STT latency, partial-result counts, and success rates.
- **Structured logging** to both console and a timestamped log file.

## Requirements

- Python 3.7+
- An Azure Speech resource (key + region)
- A test audio file in WAV (PCM) format

Install the Azure Speech SDK:

```bash
pip install azure-cognitiveservices-speech
```

## Configuration

Edit the configuration constants near the top of the script:

| Variable | Description |
|----------|-------------|
| `AZURE_SPEECH_KEY` | Your Azure Speech resource key |
| `AZURE_REGION` | Your Azure Speech resource region (e.g. `eastus`) |
| `LANGUAGE` | Recognition language (default `en-US`) |
| `AUDIO_FILE_PATH` | Path to the test WAV file (default `test_audio.wav`) |

> **Note:** Do not commit real keys. Use environment variables or a secrets manager in production.

## Usage

```bash
python azure_stt_streaming_load_test.py
```

The test runs in two phases:

1. **Phase 1 — Sequential streaming tests:** Runs a small number of sequential requests to measure baseline latency.
2. **Phase 2 — Concurrent streaming tests:** Ramps through concurrency levels `[10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]`, with wait intervals between batches.

Results are printed to the console and saved to a log file named `azure_speech_test_<timestamp>.log`.

## Metrics Explained

| Metric | Meaning |
|--------|---------|
| **TTFT** (Time to First Token) | Time from request start until the first partial recognition result. |
| **TTFR** (Time to Final Result) | Time from request start until the final recognized text. |
| **STT Latency** | Recognition latency measured in milliseconds. |
| **Success rate** | Percentage of streams that completed successfully. |

## References

- [Azure AI Speech documentation](https://learn.microsoft.com/azure/ai-services/speech-service/)
- [Speech to text quickstart](https://learn.microsoft.com/azure/ai-services/speech-service/get-started-speech-to-text)
- [Azure Speech SDK for Python](https://learn.microsoft.com/python/api/overview/azure/cognitiveservices-speech-readme)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## License Summary

This sample code is provided under the MIT-0 license. See the LICENSE file.
