from transformers import AutoFeatureExtractor
from onnxruntime import InferenceSession
from datasets import load_dataset
import numpy as np
import time

# load image
dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]

# load model
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
session = InferenceSession("../models/resnet-50/onnx/model_cls.onnx")

start_time = time.time()
# ONNX Runtime expects NumPy arrays as input
inputs = feature_extractor(image, return_tensors="np")	
outputs = session.run(output_names=["logits"], input_feed=dict(inputs))

end_time = time.time()
elapsed_time_ms = (end_time - start_time) * 1000
print(f"Inference completed in {elapsed_time_ms:.2f} milliseconds")

# Get the logits from the outputs
logits = outputs[0]

# Get the predicted class index
predicted_class_idx = np.argmax(logits, axis=-1).item()

# Load ImageNet class labels
def load_imagenet_labels():
    with open("../models/resnet-50/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

imagenet_class_labels = load_imagenet_labels()

# Get the predicted class name
predicted_class_name = imagenet_class_labels[predicted_class_idx]

# Print the predicted class
print(f"Predicted class: {predicted_class_name}")

