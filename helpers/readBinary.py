import struct

def read_detection(f):
    # read classLabel (size_t → assume 8 bytes)
    class_label = struct.unpack("<Q", f.read(8))[0]

    objectness = struct.unpack("<d", f.read(8))[0]

    bbox = struct.unpack("<dddd", f.read(32))  # x, y, w, h

    contour_size = struct.unpack("<Q", f.read(8))[0]

    contour = []
    for _ in range(contour_size):
        x, y = struct.unpack("<dd", f.read(16))
        contour.append((x, y))

    return {
        "class_label": class_label,
        "objectness": objectness,
        "bbox": bbox,
        "contour": contour
    }

filepath = "assets/dummy_results_jpeg/hamburg_000000_014030_leftImg8bit_detection_0.bin"

with open(filepath, 'rb') as file:
    print(read_detection(file))