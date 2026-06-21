# onnx_graph_modify.py
import hashlib
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import sys

if __name__ == "__main__":
    # ---- constants from your export ----
    B = 1           # batch
    N = 300          # max detections (post-NMS)
    P = 32           # mask prototype channels
    H = 128
    W = 256
    HW = H * W       # 32768

    SRC = sys.argv[1]
    DST = f"{SRC[:-5]}_modified_fp32.onnx"

    # ---- Load and import
    model = onnx.load(SRC)
    graph = gs.import_onnx(model)

    # ---- Locate original outputs
    # Expect: output0 [B,N,38] = [x1,y1,x2,y2, conf, classLabelId, coeffs(32)]
    #         output1 [B,P,H,W] = prototypes
    output0 = next(t for t in graph.outputs if "output0" in t.name)
    output1 = next(t for t in graph.outputs if "output1" in t.name)

    # ---- 1) Split output0 into fixed chunks (hard 32 for coeffs)
    boxes = gs.Variable("boxes", dtype=output0.dtype, shape=[B, N, 4])
    objectness = gs.Variable("objectness", dtype=output0.dtype, shape=[B, N, 1])
    classLabel = gs.Variable("classlabel", dtype=output0.dtype, shape=[B, N, 1])
    coeffs32 = gs.Variable("coeffs32", dtype=output0.dtype, shape=[B, N, P])

    split_sizes = gs.Constant(
        "split_sizes_o0",
        np.array([4, 1, 1, P], dtype=np.int64)
    )

    graph.nodes.append(gs.Node(
        op="Split",
        name="node_of_split_o0",
        inputs=[output0, split_sizes],
        outputs=[boxes, objectness, classLabel, coeffs32],
        attrs={"axis": 2},
    ))

    # ---- 2) Reshape prototypes to [B,P,HW] using a CONSTANT shape (TRT-friendly)
    P_flat = gs.Variable("P_flat", dtype=output1.dtype, shape=[B, P, HW])
    graph.nodes.append(gs.Node(
        op="Reshape",
        name="node_of_P_flat",
        inputs=[output1, gs.Constant("shape_b_p_hw", np.array([B, P, HW], dtype=np.int64))],
        outputs=[P_flat],
    ))
    
    P_flat_cast = gs.Variable("P_flat_cast", dtype=output0.dtype, shape=[B, P, HW])
    
    graph.nodes.append(gs.Node(
        op="Cast",
        name="node_of_pflat_cast",
        inputs=[P_flat],
        outputs=[P_flat_cast],
        attrs={"to": onnx.TensorProto.FLOAT}
    ))

    # ---- 3) MatMul: [B,N,P] x [B,P,HW] -> [B,N,HW]
    Y = gs.Variable("Y", dtype=output0.dtype, shape=[B, N, HW])
    graph.nodes.append(gs.Node(
        op="MatMul",
        name="node_of_Y",
        inputs=[coeffs32, P_flat_cast],
        outputs=[Y],
    ))
    
    Y_sigmoid = gs.Variable("Y_sigmoid", dtype=output0.dtype, shape=[B, N, HW])
    graph.nodes.append(gs.Node(
        op="Sigmoid",
        name="node_of_Y_sigmoid",
        inputs=[Y],
        outputs=[Y_sigmoid],
    ))

    # ---- 4) Reshape to masks [B,N,H,W] using a CONSTANT shape
    masks = gs.Variable("masks", dtype=output0.dtype, shape=[B, N, H, W])
    graph.nodes.append(gs.Node(
        op="Reshape",
        name="node_of_masks",
        inputs=[Y_sigmoid, gs.Constant("shape_b_n_h_w", np.array([B, N, H, W], dtype=np.int64))],
        outputs=[masks],
    ))

    # ---- Replace outputs (drop originals to save bandwidth)
    graph.outputs = [boxes, classLabel, objectness, masks]

    # ---- Export + (optional) shape inference for nicer value_info
    mo = gs.export_onnx(graph)
    mo = shape_inference.infer_shapes(mo)
    onnx.save(mo, DST)

    # ---- Print checksum + key shapes so you can verify you're building same bytes
    data = open(DST, "rb").read()
    print("SAVED:", DST)
    print("SHA256:", hashlib.sha256(data).hexdigest())

    def vi_shape(g, name):
        for vi in list(g.value_info) + list(g.input) + list(g.output):
            if vi.name == name and vi.type.tensor_type.HasField("shape"):
                dims = vi.type.tensor_type.shape.dim
                return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims]
        return None

    gg = mo.graph
    print("boxes    :", vi_shape(gg, "boxes"))
    print("classLabel    :", vi_shape(gg, "classlabel"))
    print("objectness     :", vi_shape(gg, "objectness"))
    print("coeffs32 :", vi_shape(gg, "coeffs32"))
    print("P_flat   :", vi_shape(gg, "P_flat"))
    print("Y        :", vi_shape(gg, "Y"))
    print("masks    :", vi_shape(gg, "masks"))