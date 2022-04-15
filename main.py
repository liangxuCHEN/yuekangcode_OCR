import paddlehub as hub
import cv2
import os
import argparse
import pandas as pd
import numpy as np


def file_process(file_path):
    img_path_list = []
    name_list = []
    for _, _, filenames in os.walk(file_path):
        for img_name in filenames:
            img_name = img_name.split('.')[0]
            img_name = img_name.split('-')[0]
            name_list.append(img_name)

    for parent, dirnames, filenames in os.walk(args.image_path):
        for img in filenames:
            img_path_list.append(os.path.join(args.image_path, img))

    return name_list, img_path_list


def get_ocr_data(test_img_path):
    # 加载移动端预训练模型
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")  # 移动端级别
    # 读取照片路径
    np_images = [cv2.imread(image_path) for image_path in test_img_path]

    return ocr.recognize_text(
        images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
        use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
        visualization=False,  # 是否将识别结果保存为图片文件；
        box_thresh=0.6,  # 检测文本框置信度的阈值；
        text_thresh=0.7)  # 识别中文文本置信度的阈值；


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-p', type=str, default='./image')
    parser.add_argument('--out_path', '-o', type=str, default='./output')
    parser.add_argument('--image_name', '-n', type=str, default='')
    args = parser.parse_args()

    if not args.image_name == '':
        img_path = os.path.join(args.image_path, args.image_name)
        path_list = [img_path]
        name_list = [args.image_name.split(".").split("-")[0]]
    else:
        name_list, path_list = file_process(args.image_path)
        print(name_list, len(name_list))

    results = get_ocr_data(path_list)

    total_result = []
    for idx, res in enumerate(results):
        person_res = [name_list[idx]]
        is_72 = False
        data = res['data']
        begin_id = 1000
        for info_idx, infomation in enumerate(data):
            if infomation['text'] == "粤康码" and begin_id == 1000:
                begin_id = info_idx

            if info_idx == begin_id + 2:
                # 姓名
                person_res.append(infomation['text'])
            elif info_idx == begin_id + 4:
                # 截图时间
                person_res.append(infomation['text'])
            elif info_idx == begin_id + 11:
                # 核酸检测
                # 72小时
                try:
                    time_checked = int(infomation['text'])
                    person_res.append(f"阴性{time_checked}")
                    is_72 = True
                except Exception as e:
                    person_res.append(infomation['text'])

            elif info_idx == begin_id+13 and not is_72:
                # 核酸检测时间
                person_res.append(infomation['text'])
            elif info_idx == begin_id+15 and is_72:
                person_res.append(infomation['text'])

        total_result.append(person_res)

    total_result = np.array(total_result)
    df = pd.DataFrame({
        '提交姓名': total_result[:, 0],
        '图片姓名': total_result[:, 1],
        '截图时间': total_result[:, 2],
        '采样时间': total_result[:, 4],
        '检测结果': total_result[:, 3]
    })
    print(df.head())
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    df.to_excel(os.path.join(args.out_path, 'output.xlsx'))