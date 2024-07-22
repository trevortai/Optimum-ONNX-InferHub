# Optimum-InferHub
Using Optimum to run inference on a variety of execution providers for ONNX models

Status: only working with DML right now on windows

To start, you will need an onnx model.

It is also recommended to work in conda environments, either install miniconda or anaconda to your system.

Start the anaconda terminal and run the following to create an environment:

```conda create -n ort python```

```conda activate ort```

Convert models to onnx with optimum: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model

Basic sample:

```pip install optimum```

```optimum-cli export onnx --model distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/```

If you do not have one, you can find models from hugging face and run the convert script from optimum on it if it is not already onnx.

For Whisper Example:

  ```cd whisper```

  ```conda install ffmpeg```
  
  ```pip install -r requirements.txt```
  
  ```python read.py with --model localmodel --audio audiofile --timer```
