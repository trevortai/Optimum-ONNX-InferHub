from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
import onnxruntime as ort
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from init_args import parse_args_classification
import time


def main():
    # Parse arguments
    args = parse_args_classification()

    # Get the available execution providers
    available_providers = ort.get_available_providers()
    print("Available execution providers:", available_providers)


    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ORTModelForSequenceClassification.from_pretrained(args.model, provider="DmlExecutionProvider", use_cache=False)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    pred = classifier(args.text)
    print(pred)

if __name__ == "__main__":
    main()