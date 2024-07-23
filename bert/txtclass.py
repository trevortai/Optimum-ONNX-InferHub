from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
import onnxruntime as ort
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from init import parse_args_classification, EPselect
import time


def main():
    # Parse arguments
    args = parse_args_classification()

    # Get the available execution providers
    available_providers = ort.get_available_providers()
    print("Available execution providers:", available_providers)


    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ORTModelForSequenceClassification.from_pretrained(args.model, provider=EPselect(args, available_providers), use_cache=False)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Measure the time taken for inference if the timer argument is provided
    if args.timer:
        start_time = time.time()

    # Perform inference using the provided input path
    pred = classifier(args.text)

    if args.timer:
        end_time = time.time()
        print(f"Inference completed in {end_time - start_time:.2f} seconds")

    print(pred)

if __name__ == "__main__":
    main()