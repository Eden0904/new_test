from mmocr.apis import TextDetInferencer, TextRecInferencer
import cv2
import matplotlib.pyplot as plt

# 添加编写好的组件
from tools import imgdata_processs
from tools import visualize

def ocr_model(det_imgs):
    '''

    :param det_imgs: img_list = [[nparry1], [nparry2]]
    :return: 一个包含识别结果的字典
    '''
    det_model = '/root/workspace/ocr_models/config/dbnet_resnet18_fpnc_1200e_icdar2015.py'
    det_path = '/root/workspace/ocr_models/config/epoch_160.pth'
    rec_model = '/root/workspace/ocr_models/config/sar_resnet31_parallel-decoder_5e_toy_evaluation.py'
    rec_path = '/root/workspace/ocr_models/config/epoch_10.pth'
    det_inferencer = TextDetInferencer(model=det_model, weights=det_path, device='cpu')
    rec_inferencer = TextRecInferencer(model=rec_model, weights=rec_path, device='cpu')

    for det_img in det_imgs:
        det_result = det_inferencer(det_img, return_datasamples = True)
        det_data_sample = det_result.get('predictions')
        pred_instances = det_data_sample[0].pred_instances
        polygons = pred_instances.polygons
        num_inputs = 0
        rec_inputs = []
        rec_inputs_per_image = []
        if polygons:
            for polygon in polygons:
                quad = imgdata_processs.bbox2poly(imgdata_processs.poly2bbox(polygon)).tolist()
                rec_inputs.append(imgdata_processs.crop_img(det_img[0], quad))
                num_inputs += 1
        rec_inputs_per_image.append(num_inputs)

        recog_results = [rec_inferencer(rec_inputs)]
        for recog_result in recog_results:
            recog_preds = recog_result.get('predictions')
            recog_items = [recog_pred.get('text') for recog_pred in recog_preds]

        recog_results_per_image = []
        start_idx = 0
        for num_inputs in rec_inputs_per_image:
            end_idx = start_idx + num_inputs
            recog_results_per_image.append(recog_items[start_idx:end_idx])
            start_idx = end_idx

        recog_result_dict = {}
        for idx, recog_results in enumerate(recog_results_per_image):
            recog_result_dict[idx + 1] = recog_results

        # # 可视化结果：如果show就显示结果，如果save_results就会把可视化好的图片返回，可以用于前端展示（目前这个还没调好，根据前端需要继续改吧）
        imgs = visualize.visualize_single_img(det_imgs, [det_data_sample], recog_results_per_image)


    return recog_result_dict, imgs



# det_image_paths = [r'E:\SIMENS_internship\ocr_test\data\273.jpg']
#
# img_lists = []
# for img_path in det_image_paths:
#     img_np = [cv2.imread(img_path)]
#     img_lists.append(img_np)
# # 模型推理
# result, imgs = ocr_model(img_lists)
#
# # 可视化
# for img in imgs:
#     plt.imshow(img)
#     plt.show()
#
# print(result)
# print(imgs)
