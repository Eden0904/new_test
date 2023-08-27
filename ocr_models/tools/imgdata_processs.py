import os
import numpy as np
from typing import Any, Callable, Optional, Type, Union, Dict, List, Sequence
from collections import abc
# from mmocr.utils.typing_utils import ArrayLike

ArrayLike = 'ArrayLike'
if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion('1.20.0'):
    from numpy.typing import ArrayLike as NP_ARRAY_LIKE
    ArrayLike = NP_ARRAY_LIKE


def poly2bbox(polygon: ArrayLike) -> np.array:
    """将多边形转换为边界框。
    Args:
         polygon (ArrayLike): 多边形的表示。可以是各种形式的对象，可以转换为一个一维的numpy数组。
             例如 list[float]、np.ndarray 或者 torch.Tensor。多边形的表示方式为 [x1, y1, x2, y2, ...]。

     Returns:
         np.array: 转换后的边界框 [x1, y1, x2, y2]
         通过计算 x 坐标的最小值、y 坐标的最小值、x 坐标的最大值和 y 坐标的最大值，创建一个新的数组，
         即转置后的边界框。数组的顺序为 [x1, y1, x2, y2]，其中 (x1, y1) 是左上角的坐标，(x2, y2) 是右下角的坐标。
    """
    assert len(polygon) % 2 == 0
    polygon = np.array(polygon, dtype=np.float32)
    x = polygon[::2]
    y = polygon[1::2]
    return np.array([min(x), min(y), max(x), max(y)])

def bbox2poly(bbox: ArrayLike, mode: str = 'xyxy') -> np.array:
    """将边界框转换为多边形。
    Args:
        bbox (ArrayLike): 边界框的表示。可以是各种形式的对象，可以通过一维索引进行访问。
         例如 list[float]、np.ndarray 或者 torch.Tensor。边界框的表示方式为 [x1, y1, x2, y2]。
        mode (str): 指定边界框的格式。可以是 'xyxy' 或 'xywh'。默认为 'xyxy'。

    Returns:
        np.array: 转换后的多边形 [x1, y1, x2, y1, x2, y2, x1, y2]。
    """
    assert len(bbox) == 4
    if mode == 'xyxy':
        x1, y1, x2, y2 = bbox
        poly = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
    elif mode == 'xywh':
        x, y, w, h = bbox
        poly = np.array([x, y, x + w, y, x + w, y + h, x, y + h])
    else:
        raise NotImplementedError('Not supported mode.')

    return poly


def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """检查序列是否为某种类型的序列。

    Args:
        seq (Sequence): 要检查的序列。
        expected_type (type or tuple): 期望的序列项类型。
        seq_type (type, optional): 期望的序列类型。默认为 None。

    Returns:
        bool: 如果 seq 有效则返回 True，否则返回 False。

    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def crop_img(src_img, box, long_edge_pad_ratio=0.4, short_edge_pad_ratio=0.2):
    """根据略微填充的边界框裁剪文本区域。
    假设边界框是一个四边形，并紧密地包围文本区域。

    Args:
        src_img (np.array): 原始图像。
        box (list[float | int]): 四边形的点。
        long_edge_pad_ratio (float): 长边填充的比例。填充长度将是短边的长度 * long_edge_pad_ratio。
            默认为 0.4。
        short_edge_pad_ratio (float): 短边填充的比例。填充长度将是长边的长度 * short_edge_pad_ratio。
            默认为 0.2。

    Returns:
        np.array: 裁剪后的图像。
    """
    assert is_seq_of(box, (float, int))
    assert len(box) == 8
    assert 0. <= long_edge_pad_ratio < 1.0
    assert 0. <= short_edge_pad_ratio < 1.0

    h, w = src_img.shape[:2]
    points_x = np.clip(np.array(box[0::2]), 0, w)
    points_y = np.clip(np.array(box[1::2]), 0, h)

    box_width = np.max(points_x) - np.min(points_x)
    box_height = np.max(points_y) - np.min(points_y)
    shorter_size = min(box_height, box_width)

    if box_height < box_width:
        horizontal_pad = long_edge_pad_ratio * shorter_size
        vertical_pad = short_edge_pad_ratio * shorter_size
    else:
        horizontal_pad = short_edge_pad_ratio * shorter_size
        vertical_pad = long_edge_pad_ratio * shorter_size

    left = np.clip(int(np.min(points_x) - horizontal_pad), 0, w)
    top = np.clip(int(np.min(points_y) - vertical_pad), 0, h)
    right = np.clip(int(np.max(points_x) + horizontal_pad), 0, w)
    bottom = np.clip(int(np.max(points_y) + vertical_pad), 0, h)

    dst_img = src_img[top:bottom, left:right]

    return dst_img

def convert_inputs(input_paths: Union[str, List[str]]) -> List[str]:
    image_paths = []

    if isinstance(input_paths, str):  # 输入为单个路径
        input_paths = [input_paths]

    for input_path in input_paths:
        if os.path.isfile(input_path):  # 输入为单个文件路径
            image_paths.append(input_path)
        elif os.path.isdir(input_path):  # 输入为文件夹路径
            for file_name in os.listdir(input_path):
                file_path = os.path.join(input_path, file_name)
                if os.path.isfile(file_path):
                    image_paths.append(file_path)

    return image_paths
