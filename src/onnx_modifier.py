"""Modify the ONNX graph to expose intermediate layer output (block39/conv2/mixin)."""

import onnx
from onnx import TensorProto, helper


def add_intermediate_output(
    input_path: str,
    output_path: str,
    layer_name: str = "/block39/conv2/mixin",
) -> None:
    """Add an intermediate layer as an additional output to the ONNX model."""
    model = onnx.load(input_path)
    graph = model.graph

    # Create a new output for the intermediate layer
    intermediate_output = helper.make_tensor_value_info(
        layer_name, TensorProto.FLOAT, None  # shape will be inferred
    )
    graph.output.append(intermediate_output)

    onnx.save(model, output_path)
    print(f"Saved modified model to {output_path}")
    print(f"Added intermediate output: {layer_name}")

    # Verify
    model2 = onnx.load(output_path)
    print(f"Outputs: {[o.name for o in model2.graph.output]}")


if __name__ == "__main__":
    add_intermediate_output(
        "/Users/ol/Documents/chessy/models/t78.onnx",
        "/Users/ol/Documents/chessy/models/t78_with_intermediate.onnx",
    )
