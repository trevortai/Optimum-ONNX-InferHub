import onnx
from onnxconverter_common import float16
from init_args import parse_args_conversion

def main():
    args = parse_args_conversion()

    # Load the model
    model = onnx.load(args.model)

    # Perform the conversion based on the direction
    if args.direction == "fp16_to_fp32":
        # Convert the model from FP16 to FP32
        print("This function is a WIP")
    elif args.direction == "fp32_to_fp16":
        # Convert the model from FP32 to FP16
        model_converted = float16.convert_float_to_float16(model, keep_io_types=True)
        output_path = args.model.replace(".onnx", "_fp16.onnx")

    # Save the converted model
    onnx.save(model_converted, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main() 