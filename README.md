# Optimum-InferHub
Using Optimum to run inference on a variety of execution providers for ONNX models

**Status:** only working with DML and CPU right now on windows, will try to include CUDA, and ROCm + Linux in the future

To start, you will need an onnx model.

It is also recommended to work in conda environments, either install miniconda or anaconda to your system.

Start the anaconda terminal and run the following to create an environment:

```conda create -n ort python```

```conda activate ort```

Convert models to onnx optimum: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model

Models that should work out of the box: https://huggingface.co/models?sort=trending&search=Optimum%2F

Basic sample:

```pip install optimum```

```optimum-cli export onnx --model distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/```

If you do not have one, you can find models from hugging face and run the convert script from optimum on it if it is not already onnx.

For Whisper Example:

  ```cd whisper```

  ```conda install ffmpeg```
  
  ```pip install -r requirements.txt```
  
  ```python read.py with --model localmodel --audio audiofile --ep dml --timer```

  dtconv-ort can be used to convert ONNX Optimum models form FP32 to FP16

  Example:
  ```python dtconv-ort --model localmodel --direction fp32_to_fp16```
  The resulting .onnx model will be placed in the same folder as the localmodel
  Generally, all other files can work with both fp16 and the original fp32, but only 1 can be in the folder to work correctly
