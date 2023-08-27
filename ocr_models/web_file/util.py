"""Utilities
"""
import re
import base64

import numpy as np
import cv2

from PIL import Image
from io import BytesIO


def base64_to_array(img_base64):
    """
    Convert base64 image data to PIL image
    """
    # 将Base64数据中的前缀剔除，只留下实际的Base64数据
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)

    # 解码为二进制数据
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))

    image_nparray = np.array(pil_image)

    bgr_array = cv2.cvtColor(image_nparray, cv2.COLOR_RGB2BGR)

    return bgr_array


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

