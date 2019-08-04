import numpy as np
import cv2
import pathlib
import os

import sys

sys.path.append("ocr/src")

from ocr import words, characters
from ocr.tfhelpers import Model

MODEL_LOC = "./ocr/models/char-clas/en/CharClassifier"
DATA_PATH_1_3 = "data/paper_1_3"
DATA_PATH_1_4 = "data/paper_1_4"
DATA_PATH_1_5 = "data/paper_1_5"
DATA_PATH_1_3_2 = "data/paper_1_3_2"
DATA_PATH_1_4_2 = "data/paper_1_4_2"
DATA_PATH_1_5_2 = "data/paper_1_5_2"

O_HEIGHT = 96
O_WIDTH = 256
HEIGHT = 192
WIDTH = 512
CHANNEL = 1

np.random.seed(1)

def get_image_from_file(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(WIDTH, HEIGHT))

    return img

def get_bboxes(img_path):
    img = get_image_from_file(img_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[:, :, 0][img_gray > 180] = 255
    img[:, :, 0][img_gray <= 180] = 0
    img[:, :, 1][img_gray > 180] = 255
    img[:, :, 1][img_gray <= 180] = 0
    img[:, :, 2][img_gray > 180] = 255
    img[:, :, 2][img_gray <= 180] = 0
    
    bboxes = words.detection(img)
    return bboxes


def preprocess(empty_img_path, img_path):
    bboxes = get_bboxes(empty_img_path)

    img = get_image_from_file(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img[img > 180] = 255
    img[img <= 180] = 0

    for bx in bboxes:
        x1, y1, x2, y2 = bx
        img[0:256, 0:x2+5] = 255

    img_rgb = np.zeros((HEIGHT, WIDTH, 3))
    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img

    bboxes = words.detection(img_rgb)

    xx1, yy1, xx2, yy2 = np.inf, np.inf, 0, 0

    for bx in bboxes:
        x1, y1, x2, y2 = bx
        if x1 < xx1:
            xx1 = x1
        if y1 < yy1:
            yy1 = y1
        if x2 > xx2:
            xx2 = x2
        if y2 > yy2:
            yy2 = y2

    img = cv2.resize(img[yy1:yy2, xx1:xx2], dsize=(O_WIDTH, O_HEIGHT))
    img = img.astype(np.float32) / 255
    img[img > 0.75] = 1
    img[img <= 0.75] = 0

    return img


def get_data_loader(question, batch_size, train=True):
    if train:
        settype = "/train"
    else:
        settype = "/test"

    if question == "3":
        path = DATA_PATH_1_3
    elif question == "4":
        path = DATA_PATH_1_4
    elif question == "5":
        path = DATA_PATH_1_5
    elif question == "3_2":
        path = DATA_PATH_1_3_2
    elif question == "4_2":
        path = DATA_PATH_1_4_2
    elif question == "5_2":
        path = DATA_PATH_1_5_2
    else:
        print("INVALID question number. Only allowed either 3, 4, 5")
        return

    empty = []
    pos = []
    neg = []

    for f in pathlib.Path(os.path.join(path, "empty").replace("\\", "/")).glob("*.png"):
        empty.append(str(f))
    for f in pathlib.Path(os.path.join(path + settype, "o").replace("\\", "/")).glob("*.png"):
        pos.append(str(f))
    for f in pathlib.Path(os.path.join(path + settype, "x").replace("\\", "/")).glob("*.png"):
        neg.append(str(f))

    for b in range(10):

        X_src_batch = np.zeros((batch_size, O_HEIGHT, O_WIDTH, CHANNEL))
        X_pos_batch = np.zeros((batch_size, O_HEIGHT, O_WIDTH, CHANNEL))
        X_neg_batch = np.zeros((batch_size, O_HEIGHT, O_WIDTH, CHANNEL))

        for i in range(batch_size):
            src_index = np.random.randint(len(pos))
            pos_index = np.random.randint(len(pos))
            neg_index = np.random.randint(len(neg))

            X_src_batch[i, :, :, 0] = preprocess(empty[0], pos[src_index])
            X_pos_batch[i, :, :, 0] = preprocess(empty[0], pos[pos_index])
            X_neg_batch[i, :, :, 0] = preprocess(empty[0], neg[neg_index])

        yield X_src_batch, X_pos_batch, X_neg_batch

