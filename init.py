import argparse
import sys

def parse_args_whisper():
    parser = argparse.ArgumentParser(description="Run Whisper model on an audio file.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--timer", action="store_true", help="Measure the time taken for inference")
    parser.add_argument("--ep", type=str, required=False, help="Execution provider to use")
    return parser.parse_args()

def parse_args_conversion():
    parser = argparse.ArgumentParser(description="Convert ONNX models between FP16 and FP32.")
    parser.add_argument("--model", type=str, required=True, help="Path to the input ONNX model")
    parser.add_argument("--direction", type=str, choices=["fp16_to_fp32", "fp32_to_fp16"], required=True, help="Conversion direction: fp16_to_fp32 or fp32_to_fp16")
    return parser.parse_args()

def parse_args_classification():
    parser = argparse.ArgumentParser(description="Run a text classification model.")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--timer", action="store_true", help="Measure the time taken for inference")
    parser.add_argument("--ep", type=str, required=False, help="Execution provider to use")
    return parser.parse_args()

def EPselect(args, available_providers):
    # Validate the execution provider argument
    if args.ep and args.ep.lower() not in ["dml", "cpu"]:
        print("Error: execution provider must be either 'dml' or 'cpu'.")
        sys.exit(1)

    # Check if the selected execution provider is available
    if args.ep.lower() == "dml" and "DmlExecutionProvider" in available_providers:
        print("Using DmlExecutionProvider for execution.")
        return "DmlExecutionProvider"
    elif args.ep.lower() == "cpu" and "CPUExecutionProvider" in available_providers:
        print("Using CPUExecutionProvider for execution.")
        return "CPUExecutionProvider"
    elif args.ep.lower() == "cuda" and "CUDAExecutionProvider" in available_providers:
        print("Using CUDAExecutionProvider for execution.")
        return "CUDAExecutionProvider"
    else:
        print(f"Error: The selected execution provider '{args.ep}' is not available.")
        sys.exit(1)
