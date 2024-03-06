import argparse
from time import time

import cv2
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from collections import OrderedDict

colors_list = [(0, 0, 0),
               (0, 0, 128),
               (0, 128, 0),
               (128, 0, 0),
               (128, 128, 0),
               (128, 0, 128),
               (0, 128, 128),
               (0, 0, 255),
               (0, 255, 0)
                ]


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


def sliding_predict(model, image, num_classes, flip=True):
    image_size = image.shape
    tile_size = (int(image_size[2] // 2.5), int(image_size[3] // 2.5))
    overlap = 1 / 3

    stride = ceil(tile_size[0] * (1 - overlap))

    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)  # (1,cls_num,block_h,block_w).
            if flip:
                # fliped_img = padded_img.flip(-1)
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions
    return total_predictions


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())  # 缩放后预测整图，再缩放到原始尺度。

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)  # 多个尺度下取平均值
    return total_predictions


def save_images(image, mask, output_path, image_file, palette):
    # Saves the image, the model output and the results after the post processing
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file + '.png'))
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))


def save_images_fp_fn(image, prediction, output, img_file, colors_list):
    image = np.asarray(image, dtype=np.float32)
    orig_image = image.copy()
    class_num = prediction.shape[0]

    base_name = os.path.basename(img_file)
    file_ext = base_name.split(".")[-1]
    json_path = img_file.replace(file_ext, "json")
    if os.path.exists(json_path):
        is_true_ng = True
    else:
        is_true_ng = False

    save_fn_path = os.path.join(output, "fn")
    save_fp_path = os.path.join(output, "fp")
    os.makedirs(save_fn_path, exist_ok=True)
    os.makedirs(save_fp_path, exist_ok=True)
    prediction_ng = False
    for i in range(1, class_num):
        mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        mask[prediction == i] = 255
        # cv2.namedWindow("mask", cv2.WINDOW_NORMAL), cv2.imshow("mask", mask), cv2.waitKey()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            prediction_ng = True
            cv2.drawContours(image, cnts, -1, colors_list[i], 3)
        # cv2.namedWindow("image", cv2.WINDOW_NORMAL), cv2.imshow("image", np.uint8(image)), cv2.waitKey()

    # fn
    if is_true_ng and prediction_ng is False:
        save_full_path = os.path.join(save_fn_path, base_name)
    # fp
    elif is_true_ng is False and prediction_ng:
        save_full_path = os.path.join(save_fp_path, base_name)
    else:
        save_full_path = os.path.join(output, base_name)
    total_image = cv2.hconcat([orig_image, image])
    cv2.imwrite(save_full_path, total_image)


def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K', 'DeepScene', 'Defect']

    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
            # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    output = args.images + "_result"
    os.makedirs(output, exist_ok=True)
    image_files = []
    for root_, dirs_, files in os.walk(args.images):
        for file_name in files:
            file_ex = file_name.split(".")[-1]
            if file_ex in ["png", "jpg", "jpeg", "bmp", "tif"]:
                image_path = os.path.join(root_, file_name)
                image_files.append(image_path)
    input_size = config['train_loader']['args']['base_size']
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        for img_file in tbar:
            start_time = time()
            image = np.asarray(Image.open(img_file).convert('RGB'), dtype=np.float32)
            image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray(np.uint8(image))
            input = normalize(to_tensor(image)).unsqueeze(0)
            prediction = model(input.to(device))
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            print("infer time: {}".format(time() - start_time))
            save_images_fp_fn(image, prediction, output, img_file, colors_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC', type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='jpg', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
