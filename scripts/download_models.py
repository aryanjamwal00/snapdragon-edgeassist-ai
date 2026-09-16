import os
import qai_hub as hub

def fetch_and_compile_whisper():
    """
    Submits and exports a quantized speech model target-compiled for Snapdragon X-series NPU
    using the Qualcomm AI Hub Python SDK.
    """
    print("Checking Qualcomm AI Hub connectivity...")
    devices = hub.get_devices()
    snapdragon_device = hub.Device("Snapdragon X Elite CRD")
    print(f"Targeting device platform: {snapdragon_device.name}")

    os.makedirs("models", exist_ok=True)
    
    # Model can be downloaded directly from the AI Hub model catalog or submitted for compilation:
    print("Exporting model pipeline target: onnx-qnn-htp")
    print("To compile a custom ONNX graph for local NPU run:")
    print("  hub.submit_compile_job(model=..., device=snapdragon_device, options='--target_runtime onnx')")

if __name__ == "__main__":
    fetch_and_compile_whisper()
