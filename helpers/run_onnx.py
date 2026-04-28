import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, pdb
from tqdm import tqdm


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def crop_mask_np(masks, boxes):
    """
    Crop masks by boxes.

    Args:
        masks: [N, H, W] mask logits or probs in proto resolution
        boxes: [N, 4] boxes in proto resolution, xyxy

    Returns:
        Cropped masks: [N, H, W]
    """
    n, h, w = masks.shape
    out = np.zeros_like(masks)

    for i in range(n):
        x1, y1, x2, y2 = boxes[i]

        x1 = max(0, min(w, int(np.floor(x1))))
        x2 = max(0, min(w, int(np.ceil(x2))))
        y1 = max(0, min(h, int(np.floor(y1))))
        y2 = max(0, min(h, int(np.ceil(y2))))

        if x2 > x1 and y2 > y1:
            out[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]

    return out


def scale_boxes_to_proto(boxes_xyxy, img_w, img_h, proto_w, proto_h):
    """
    Scale image-space xyxy boxes to proto-space xyxy boxes.
    """
    boxes = boxes_xyxy.astype(np.float32).copy()
    boxes[:, [0, 2]] *= proto_w / float(img_w)
    boxes[:, [1, 3]] *= proto_h / float(img_h)
    return boxes


def overlay_mask_on_image(image_rgb, mask, alpha=0.5):
    """
    image_rgb: [H, W, 3], uint8
    mask: [H, W], bool or {0,1}
    """
    out = image_rgb.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)  # red overlay

    mask = mask.astype(bool)
    out[mask] = ((1 - alpha) * out[mask] + alpha * color).astype(np.uint8)
    return out


def decode_nms_masks(boxes, scores, classes, masks, img_w, img_h, conf_thres=0.25, mask_thres=0.5):
    """
    Decode masks from Ultralytics ONNX export with nms=True.

    Expected:
        nms_pred:  [1, max_det, 4 + 1 + 1 + nm]
                   = [1, max_det, x1,y1,x2,y2,score,class_id,mask_coeffs...]
        nms_proto: [nm, mh, mw]
        boxes :     [max_det, 4]
        scores:     [max_det, 1]
        classes:    [max_det, 1]
        masks:      [max_det, H, W]

    Returns:
        boxes      : [K, 4] image-space xyxy
        scores     : [K]
        classes    : [K]
        mask_probs : [K, mh, mw] in proto resolution
        mask_bin   : [K, mh, mw] binary in proto resolution
    """

    keep, _ = np.where(scores > conf_thres)
    boxes = boxes[keep, :]
    scores = scores[keep, 0]
    classes = classes[keep, 0]
    masks = masks[keep, :]

    print(f"valid detections above conf {conf_thres}: {len(boxes)}")

    if len(boxes) == 0:
        return boxes, scores, classes, np.empty((0, masks.shape[1], masks.shape[2])), np.empty((0, boxes.shape[1], boxes.shape[2]), dtype=bool)

    nm, mh, mw = masks.shape
    # scale image-space boxes to proto-space for cropping
    
    boxes_proto = scale_boxes_to_proto(boxes, img_w=img_w, img_h=img_h, proto_w=mw, proto_h=mh)
    
    # binary masks
    mask_bin = masks > mask_thres

    return boxes, scores, classes, masks, mask_bin


def main():
    os.makedirs("assets/masks/no_nms_bin", exist_ok=True)
    os.makedirs("assets/masks/no_nms_prob", exist_ok=True)
    os.makedirs("assets/masks/no_nms_overlay", exist_ok=True)

    # image_path = "data/images/val/frankfurt_000001_054077_leftImg8bit.png"
    # image_path = "data/images/val/munster_000149_000019_leftImg8bit.png" # faulty case possibly
    image_path = "assets/dummy_images/hamburg_000000_014030_leftImg8bit.png"
    onnx_nms_path = "assets/onnx/last_bs1_nms_modified_fp32.onnx"

    input_w = 1024
    input_h = 512
    conf_thres = 0.25
    mask_thres = 0.5

    # ----------------------------
    # Load and preprocess image
    # ----------------------------
    im_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    im_bgr = cv2.resize(im_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    im = im_rgb.astype(np.float32) / 255.0
    im = np.transpose(im[None], (0, 3, 1, 2))  # [1, 3, H, W]

    # ----------------------------
    # Load ONNX session
    # ----------------------------
    sess_nms = ort.InferenceSession(
        onnx_nms_path,
        providers=["CPUExecutionProvider"]
    )

    inp_name = sess_nms.get_inputs()[0].name
    nms_outs = sess_nms.run(None, {inp_name: im.astype(np.float16)})

    # if len(nms_outs) != 2:
    #     raise RuntimeError(f"Expected 2 outputs from NMS ONNX, got {len(nms_outs)}")

    boxes, classes, scores, masks = nms_outs
    # ----------------------------
    # Decode masks
    # ----------------------------
    boxes, scores, classes, mask_probs, mask_bin = decode_nms_masks(
        boxes=boxes[0].astype(np.float32),
        scores=scores[0].astype(np.float32),
        classes=classes[0].astype(np.float32),
        masks=masks[0].astype(np.float32),
        img_w=input_w,
        img_h=input_h,
        conf_thres=conf_thres,
        mask_thres=mask_thres,
    )

    if len(boxes) == 0:
        print("No detections found.")
        return

    proto_h, proto_w = mask_probs.shape[1], mask_probs.shape[2]
    print("proto resolution:", (proto_h, proto_w))

    # ----------------------------
    # Save masks + overlays
    # ----------------------------
    for i in tqdm(range(len(boxes))):
        prob = mask_probs[i]
        binary = mask_bin[i].astype(np.uint8) * 255

        # Upsample masks to image resolution for easier visualization
        prob_up = cv2.resize(prob, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        bin_up = cv2.resize(binary, (input_w, input_h), interpolation=cv2.INTER_NEAREST) > 127

        # Save probability map
        plt.imsave(f"assets/masks/no_nms_prob/mask_prob_{i:03d}.png", prob_up, cmap="gray")

        # Save binary mask
        plt.imsave(f"assets/masks/no_nms_bin/mask_bin_{i:03d}.png", bin_up.astype(np.float32), cmap="gray")

        # Save overlay
        overlay = overlay_mask_on_image(im_rgb, bin_up, alpha=0.5)

        # draw bbox and text
        x1, y1, x2, y2 = boxes[i].astype(int)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            overlay_bgr,
            f"cls={int(classes[i])} conf={scores[i]:.3f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(f"assets/masks/no_nms_overlay/overlay_{i:03d}.png", overlay_bgr)

    print(f"Saved {len(boxes)} masks.")
    print("Done.")


if __name__ == "__main__":
    main()