# snapdragon-edgeassist-ai
# EdgeAssist AI: On-Device Meeting Assistant for Snapdragon® NPU

EdgeAssist AI is an entirely offline, privacy-first productivity assistant built for Snapdragon® X-Series Copilot+ PCs (such as the HP OmniBook series). It performs real-time audio transcription, speech segmentation, and meeting action-item synthesis locally by delegating model compute to the Hexagon NPU using the Qualcomm® AI Runtime (QNN) Execution Provider for ONNX Runtime.

---

## Key Features

- **100% Offline & Private:** Zero audio data, transcripts, or telemetry leave the device.
- **Hexagon NPU Accelerated:** Offloads heavy matrix operations from CPU/GPU to the Hexagon Tensor Processor (HTP) for high TOPS throughput and low thermal footprint.
- **Power Efficient:** Consumes under 5W during sustained live inference sessions.
- **Optimized Model Pipeline:** Integrates models pre-quantized and profiled via Qualcomm AI Hub.

---

## Architecture Overview
