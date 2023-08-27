import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

def visualize_results(image_paths, det_result_dict, recog_result_dict, show_image=False, save_results=False):
    # 设置要可视化的字体，默认是宋体
    fontpath = "simsun.ttc"

    results = []  # 存储可视化的图像数据

    for image_path in image_paths:
        det_results = det_result_dict[image_path]  # 获取当前图片的检测结果列表
        recog_results = recog_result_dict[image_path]  # 获取当前图片的识别结果列表

        image = cv2.imread(image_path)

        # 文字检测和识别结果可视化
        for det_item in det_results:
            polygons = det_item.pred_instances.polygons
            # 将多边形的坐标从数组转换为整数
            for polygon, recog_text in zip(polygons, recog_results):
                polygon = polygon.reshape((1, -1, 2)).astype(np.int32)
                if len(polygon) > 0:
                    x = min(polygon[0][:, 0])  # 多边形最左侧的 x 坐标
                    y = int(np.mean(polygon[0][:, 1]))  # 多边形的 y 坐标中心点

                    # 在图像上绘制文本
                    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)

                    font = ImageFont.truetype(fontpath, 15)  # 设定字体和大小

                    # 获取文本的宽度和高度
                    text_width, text_height = draw.textsize(recog_text, font=font)

                    # 绘制文本背景
                    draw.rectangle([(x, y), (x + text_width, y + text_height)], fill=(255, 255, 255, 127))

                    # 绘制文本
                    draw.text((x, y), recog_text, font=font, fill=(255, 0, 0, 255))
                    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 保存或显示每张图片的结果
        if show_image:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        if save_results:
            # 将图像数据保存到内存中
            image_data = cv2.imencode('.png', image)[1].tobytes()
            results.append(image_data)

    return results if not show_image else None

def visualize_imgs_list(image_paths, det_result_dict, recog_result_dict):
    # 设置要可视化的字体，默认是宋体
    fontpath = "simsun.ttc"

    results = []  # 存储可视化的图像数据

    for image_path in image_paths:
        det_results = det_result_dict[image_path]  # 获取当前图片的检测结果列表
        recog_results = recog_result_dict[image_path]  # 获取当前图片的识别结果列表

        image = cv2.imread(image_path)

        # 文字检测和识别结果可视化
        for det_item in det_results:
            polygons = det_item.pred_instances.polygons
            # 将多边形的坐标从数组转换为整数
            for polygon, recog_text in zip(polygons, recog_results):
                polygon = polygon.reshape((1, -1, 2)).astype(np.int32)
                if len(polygon) > 0:
                    x = min(polygon[0][:, 0])  # 多边形最左侧的 x 坐标
                    y = int(np.mean(polygon[0][:, 1]))  # 多边形的 y 坐标中心点

                    # 在图像上绘制文本
                    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)

                    font = ImageFont.truetype(fontpath, 15)  # 设定字体和大小

                    # 获取文本的宽度和高度
                    text_width, text_height = draw.textsize(recog_text, font=font)

                    # 绘制文本背景
                    draw.rectangle([(x, y), (x + text_width, y + text_height)], fill=(255, 255, 255, 127))

                    # 绘制文本
                    draw.text((x, y), recog_text, font=font, fill=(255, 0, 0, 255))
                    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # 改变颜色通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results.append(image)

    return results

def visualize_single_img(det_imgs, det_results, recog_results):
    # 设置要可视化的字体，默认是宋体
    fontpath = "simsun.ttc"

    results = []  # 存储可视化的图像数据
    for images, det_item, recog_item in zip(det_imgs, det_results, recog_results):
        image = images[0]
        polygons = det_item[0].pred_instances.polygons
        for polygon, recog_text in zip(polygons, recog_item):
            polygon = polygon.reshape((1, -1, 2)).astype(np.int32)
            if len(polygon) > 0:
                x = min(polygon[0][:, 0])  # 多边形最左侧的 x 坐标
                y = int(np.mean(polygon[0][:, 1]))  # 多边形的 y 坐标中心点

                # 在图像上绘制文本
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)

                font = ImageFont.truetype(fontpath, 15)  # 设定字体和大小

                # 获取文本的宽度和高度
                text_width, text_height = draw.textsize(recog_text, font=font)

                # 绘制文本背景
                draw.rectangle([(x, y), (x + text_width, y + text_height)], fill=(255, 255, 255, 127))

                # 绘制文本
                draw.text((x, y), recog_text, font=font, fill=(255, 0, 0, 255))
                image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 改变颜色通道
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results.append(image)

    return results