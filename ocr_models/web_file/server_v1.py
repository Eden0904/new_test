from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import json
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

import inference

from util import base64_to_array

# 创建 Flask 应用
app = Flask(__name__)
app.debug = True

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')

# 定义一个 API 路由，接收 POST 请求
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        restored_img_lists = []

        # Get the image from post request
        #目前对于单个文件进行转换
        img = [base64_to_array(request.json)]
        restored_img_lists.append(img)

        resluts, imgs = inference.ocr_model(restored_img_lists)
        # img_strings = [base64.b64encode(img).decode('utf-8') for img in imgs]

        #尝试对多个图片内进行数据转换
        img_b64 = []
        for img in imgs:

            pil_image = Image.fromarray(img)

            # 将 PIL 图像对象转换为字节流
            image_bytes = BytesIO()
            pil_image.save(image_bytes, format="PNG")  # 可以选择适当的图像格式

            # 将字节流编码为 Base64 字符串
            base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

            # 创建一个包含 Base64 编码的图像数据的 JSON 对象
            img_b64.append(base64_image)

        response = {
            'results': '{}'.format(resluts),
            'img_b64': img_b64
        }
     
        # 使用 jsonpickle 对响应进行编码，并设置 ensure_ascii 参数为 False
        response_pickled = json.dumps(response, ensure_ascii=False)

        # 返回带有响应信息的 JSON 响应对象
        '''
        Response: 这是 Flask 中用于构建响应对象的类。
        response=response_pickled: 这是响应的主体内容。
        status=200: 这是 HTTP 响应的状态码。200 表示成功处理请求，并且服务器将返回所请求的数据。HTTP 状态码用于指示服务器对请求的处理情况。
        mimetype="application/json": 这是响应的内容类型。在这个例子中，application/json 表示响应的主体内容是 JSON 格式的数据。这个信息告诉客户端如何正确解析服务器返回的数据。
        '''
        return Response(response=response_pickled, status=200, mimetype="application/json")
    
    return  None


# start flask app  启动 Flask 应用
if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5000)
    app.run()
