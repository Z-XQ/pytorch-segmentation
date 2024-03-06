import json
import os

import cv2
import numpy as np


def json_to_color_mask(json_full_path: str, class_color=None):
    json_content = load_label(json_full_path)
    image_width = json_content["imageWidth"]
    image_height = json_content["imageHeight"]
    mask = np.zeros((image_height, image_width, 3), np.uint8)
    shapes = json_content["shapes"]
    for cur_shape in shapes:
        points = cur_shape["points"]
        shape_type = cur_shape["shape_type"]
        color = class_color[cur_shape["label"]]
        if shape_type == "polygon":
            cnt = np.array(points).reshape((-1, 1, 2))
            cnt_int64 = np.int64(cnt)
            cv2.drawContours(mask, [cnt_int64], -1, color, thickness=-1)
        elif shape_type == "rectangle":
            x1y1 = points[0]
            x2y2 = points[1]
            cv2.rectangle(mask, pt1=(x1y1[0], x1y1[1]), pt2=(x2y2[0], x2y2[1]), color=color, thickness=-1)
        elif shape_type == "circle":
            center = points[0]
            pt = points[1]
            radius = np.sqrt(np.power(center[0]-pt[0], 2) + np.power(center[1]-pt[1], 2))
            cv2.circle(mask, center=center, radius=radius, color=color, thickness=-1)
        else:
            pass

    return mask


def json_to_cls_mask(json_full_path: str, class_id_dict=None):
    json_content = load_label(json_full_path)
    image_width = json_content["imageWidth"]
    image_height = json_content["imageHeight"]
    mask = np.zeros((image_height, image_width), np.int32)
    shapes = json_content["shapes"]
    for cur_shape in shapes:
        points = cur_shape["points"]
        shape_type = cur_shape["shape_type"]
        color = class_id_dict[cur_shape["label"]]
        if shape_type == "polygon":
            cnt = np.array(points).reshape((-1, 1, 2))
            cnt_int64 = np.int64(cnt)
            cv2.drawContours(mask, [cnt_int64], -1, color, thickness=-1)
        elif shape_type == "rectangle":
            x1y1 = points[0]
            x2y2 = points[1]
            cv2.rectangle(mask, pt1=(x1y1[0], x1y1[1]), pt2=(x2y2[0], x2y2[1]), color=color, thickness=-1)
        elif shape_type == "circle":
            center = points[0]
            pt = points[1]
            radius = np.sqrt(np.power(center[0] - pt[0], 2) + np.power(center[1] - pt[1], 2))
            cv2.circle(mask, center=center, radius=radius, color=color, thickness=-1)
        else:
            pass

    return mask


def load_label(json_full_path):
    json_file = open(json_full_path, encoding="utf-8", mode="r")
    json_content = json.load(json_file)
    return json_content


if __name__ == '__main__':
    data_path = r"D:\zxq\data\df\lens_base\all_ng_label"
    image_ext = ["jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"]
    # class_colors = {
    #     "glue": (0, 0, 128),
    #     "crush": (0, 128, 0),
    #     "crack": (128, 0, 0),
    #     "white_dot": (0, 128, 128),
    #     "feather": (128, 0, 128),
    #     "other": (128, 128, 0),
    # }
    class_cls = {  # background是0
        "glue": 1,
        "crush": 2,
        "crack": 3,
        "white_dot": 4,
        "feather": 5,
        "other": 6,
    }
    for root_, dirs_, files in os.walk(data_path):
        for file_name in files:
            cur_image_ext = "." + file_name.split(".")[-1]
            if cur_image_ext in image_ext:
                image_path = os.path.join(root_, file_name)
                json_path = image_path.replace(cur_image_ext, ".json")
                if os.path.isfile(json_path):
                    # mask = json_to_color_mask(json_path, class_colors)
                    mask = json_to_cls_mask(json_path, class_cls)
                    cv2.namedWindow("mask", cv2.WINDOW_NORMAL), cv2.imshow("mask", mask), cv2.waitKey()