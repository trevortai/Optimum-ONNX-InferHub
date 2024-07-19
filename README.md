# Optimum-InferHub
Using Optimum to run inference on a variety of execution providers for ONNX models


To start, you will need an onnx model

If you do not have one, you can find models from hugging face and run the convert script from optimum on it if it is not already onnx.

For Whisper Example:
  ```conda install ffmpeg```
  ```pip install -r requirements.txt```
  ```python read.py with --model localmodel --audio audiofile --timer```
