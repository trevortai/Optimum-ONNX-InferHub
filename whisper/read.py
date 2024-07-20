from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
import onnxruntime as ort
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from init_args import parse_args_whisper
import time

def main():
    # Parse arguments
    args = parse_args_whisper()

    # Get the available execution providers
    available_providers = ort.get_available_providers()
    print("Available execution providers:", available_providers)

    # Load the processor and model using the provided model name
    processor = AutoProcessor.from_pretrained(args.model)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(args.model, provider="DmlExecutionProvider", use_cache=False)
    whisper = pipeline(task="automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)

    # Measure the time taken for inference if the timer argument is provided
    if args.timer:
        start_time = time.time()

    # Perform inference using the provided input path
    read = whisper(args.audio)

    if args.timer:
        end_time = time.time()
        print(f"Inference completed in {end_time - start_time:.2f} seconds")

    print(read)

if __name__ == "__main__":
    main()