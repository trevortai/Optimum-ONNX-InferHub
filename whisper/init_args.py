import argparse

def parse_args_whisper():
    parser = argparse.ArgumentParser(description="Run Whisper model on an audio file.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--timer", action="store_true", help="Measure the time taken for inference")
    return parser.parse_args()

def parse_args_conversion():
    parser = argparse.ArgumentParser(description="Convert ONNX models between FP16 and FP32.")
    parser.add_argument("--model", type=str, required=True, help="Path to the input ONNX model")
    parser.add_argument("--direction", type=str, choices=["fp16_to_fp32", "fp32_to_fp16"], required=True, help="Conversion direction: fp16_to_fp32 or fp32_to_fp16")
    return parser.parse_args()