"""Explore the ONNX graph of Leela T78 to find intermediate layers."""

import onnx


def explore_graph(model_path: str) -> None:
    model = onnx.load(model_path)
    graph = model.graph

    print("=== INPUTS ===")
    for inp in graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")

    print("\n=== OUTPUTS ===")
    for out in graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    print(f"\n=== NODES: {len(graph.node)} total ===")

    # Find residual block outputs (typically Add nodes after SE blocks)
    residual_adds = []
    for i, node in enumerate(graph.node):
        if node.op_type == "Add":
            residual_adds.append((i, node.name, node.output[0]))

    print(f"\n=== ADD NODES (potential residual connections): {len(residual_adds)} ===")
    for idx, name, output in residual_adds:
        print(f"  [{idx:4d}] {name or '(unnamed)'} -> {output}")

    # Also look for nodes with "block" or "residual" in their names
    print("\n=== NODES with 'block'/'residual'/'squeeze' in name ===")
    for i, node in enumerate(graph.node):
        name = node.name or ""
        if any(kw in name.lower() for kw in ["block", "residual", "squeeze", "se_"]):
            print(f"  [{idx:4d}] {node.op_type}: {name} -> {node.output}")

    # Print unique op types
    op_types = set(node.op_type for node in graph.node)
    print(f"\n=== UNIQUE OP TYPES ({len(op_types)}) ===")
    for op in sorted(op_types):
        count = sum(1 for n in graph.node if n.op_type == op)
        print(f"  {op}: {count}")

    # Print all node names to find patterns
    print("\n=== FIRST 20 NODE NAMES ===")
    for i, node in enumerate(graph.node[:20]):
        print(f"  [{i:4d}] {node.op_type}: {node.name or '(unnamed)'} -> {node.output}")

    print("\n=== LAST 40 NODE NAMES ===")
    for i, node in enumerate(graph.node[-40:], len(graph.node) - 40):
        print(f"  [{i:4d}] {node.op_type}: {node.name or '(unnamed)'} -> {node.output}")


if __name__ == "__main__":
    explore_graph("/Users/ol/Documents/chessy/models/t78.onnx")
