import torch
from torch.autograd import Variable as V
import cv2
import os
import importlib
import numpy as np
from tqdm import tqdm

SOURCE_PATH = r'./dataB'
TARGET_PATH = rf'./B'
MODEL_FILENAME = r'weights/USwinNet_B.th'
MODEL_NAME = r'USwinNet_2C'


def test_one_image(net, img):
    img = np.load(SOURCE_PATH + '/' + img)
    D, H, W = img.shape
    data = []

    data.append(img[:32, :192, :])
    data.append(img[-32:, :192, :])
    data.append(img[:32, -192:, :])
    data.append(img[-32:, -192:, :])
    data.append(img[:32, 192:384, :])
    data.append(img[-32:, 192:384, :])

    final_mask = []
    for i in data:
        img2 = np.expand_dims(i, axis=0)
        img2 = np.expand_dims(img2, axis=4)
        img2 = img2.transpose(0, 4, 1, 3, 2)
        img2 = img2 * 1.0
        with torch.no_grad():
            img2 = V(torch.Tensor(img2).cuda())
            mask = net.forward(img2).squeeze().cpu().data.numpy()
            mask = mask[1, :, :, :]
            mask = mask.transpose(0, 2, 1)
        final_mask.append(mask)

    # crossline
    up = np.concatenate((final_mask[0][:, :, :], final_mask[4][:, :, :], final_mask[2][:, -144:, :]),
                        axis=1)
    down = np.concatenate((final_mask[1][:, :, :], final_mask[5][:, :, :], final_mask[3][:, -144:, :]), axis=1)

    final = np.concatenate((up[:22, :, :], down[-22:, :, :]), axis=0)
    final_data = final[0, :, :]
    for j in range(D - 1):
        final_data = np.concatenate((final_data, final[j + 1, :, :]), axis=0)

    return final_data


if __name__ == "__main__":

    image_list = list(filter(lambda x: x.find('data.npy') != -1, os.listdir(SOURCE_PATH)))
    if not os.path.exists(TARGET_PATH):
        os.mkdir(TARGET_PATH)

    net = getattr(importlib.import_module('USwinNet'), MODEL_NAME)
    net = net().cuda()
    net.load_state_dict(torch.load(MODEL_FILENAME))
    net.eval()

    pbar = tqdm(total=len(image_list))

    for i, name in enumerate(image_list):
        mask = test_one_image(net, name)
        H, W = mask.shape
        new_mask = np.zeros(shape=(H, W), dtype='uint8')
        for i in range(H):
            t = np.argmax(mask[i, :])
            new_mask[i, t] = 255
        new_mask = new_mask.T
        cv2.imwrite(f'{os.path.join(TARGET_PATH, name[:-4])}.png', new_mask.astype(np.uint8))
        pbar.update(1)

    pbar.close()
