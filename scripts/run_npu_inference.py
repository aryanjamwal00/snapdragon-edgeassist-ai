import os
import sys
import time
import numpy as np
import onnxruntime as ort

def create_qnn_session(model_path: str) -> ort.InferenceSession:
    """
    Initializes an ONNX Runtime session leveraging the Qualcomm Hexagon NPU 
    via the QNN Execution Provider (HTP backend).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Set up session options
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # QNN HTP (Hexagon Tensor Processor) configuration options
    # On Windows ARM64, 'QnnHtp.dll' routes execution to the NPU
    qnn_options = {
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "burst",          # Options: low_power, sustained_high_performance, burst
        "enable_htp_fp16_precision": "1",          # Enable FP16/INT4 NPU math operations
    }

    providers = [
        ("QNNExecutionProvider", qnn_options),
        "CPUExecutionProvider"                     # Fallback to ARM64 CPU if needed
    ]

    print(f"Loading model on Snapdragon NPU: {model_path}...")
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers
    )

    active_providers = session.get_providers()
    print(f"Active Execution Providers: {active_providers}")
    if "QNNExecutionProvider" not in active_providers:
        print("[WARNING] QNNExecutionProvider was not loaded. Running on fallback provider.")
    else:
        print("[SUCCESS] Successfully accelerated on Snapdragon Hexagon NPU (HTP).")

    return session

def run_benchmark(session: ort.InferenceSession, iterations: int = 10):
    input_details = session.get_inputs()[0]
    input_name = input_details.name
    input_shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_details.shape]
    input_type = np.float32

    dummy_input = np.random.randn(*input_shape).astype(input_type)

    print(f"Warm-up run with shape {input_shape}...")
    session.run(None, {input_name: dummy_input})

    print(f"Benchmarking inference across {iterations} iterations on NPU...")
    latencies = []
    for i in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    avg_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    print(f"Average Inference Latency: {avg_ms:.2f} ms (+/- {std_ms:.2f} ms)")

if __name__ == "__main__":
    sample_model = sys.argv[1] if len(sys.argv) > 1 else "models/sample_model.onnx"
    try:
        sess = create_qnn_session(sample_model)
        run_benchmark(sess)
    except Exception as e:
        print(f"Error during NPU execution: {e}")
