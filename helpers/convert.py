import cv2
from glob import glob
import sys, os
from tqdm import tqdm

dir = sys.argv[1]
files = glob(f"{dir}/*")
save_dir = f"{dir}_jpeg"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for filepath in tqdm(files, total=len(files)):
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)

    filename = os.path.basename(filepath)
    root, ext = os.path.splitext(filename)
    newfilename = os.path.join(save_dir, f"{root}.jpg")
    cv2.imwrite(newfilename, image, [cv2.IMWRITE_JPEG_QUALITY, 95])