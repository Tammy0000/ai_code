import os
import random
import time
import torch
from PIL import Image
from torch import nn, functional
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from captcha.image import ImageCaptcha
from PIL import Image

tmp_number = [chr(i) for i in range(48, 58)]
tmp_string = [chr(i) for i in range(97, 123)]
ver_code = tmp_number + tmp_string
ver_dict = {}
_tmp_list = []
_file = None
_num = 0
for i in ver_code:
    ver_dict[_num] = i
    _num += 1

model_path = "D:\\AI_about\\vercode_16_0.000534.pth"
Paths_ts = "D:\\AI_about\\data_test"


def test_model(paths=model_path, path_test=Paths_ts):
    model = torch.load(paths)
    model.eval()
    model.cuda()
    torch.no_grad()
    trains = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])
    Lst = os.listdir(path_test)
    acc_list = []
    for out_i in Lst:
        Img = Image.open(os.path.join(path_test, out_i))
        Img = trains(Img).cuda()
        Iu = Img.unsqueeze(0)
        img_model = model(Iu).cuda()
        str = ""
        for i in img_model.split(36, 1):
            out = torch.max(i, 1)
            out = out[1].item()
            str = str + ver_dict[out]
        if str == out_i.split(".")[0]:
            acc_list.append(str)
    acc = round(len(acc_list) / len(Lst) * 100, 2)
    print(f"Accuracy: {acc}%")
    # for i in acc_list:
    #     print(i)
    return acc


# 生成验证码模块
def create_test(in_num, code_num=5):
    if len(os.listdir(Paths_ts)) > 0:
        os.popen(r"del D:\AI_about\data_test\*.jpg")
    while len(_tmp_list) < in_num:
        _nq = 0
        str = ""
        while _nq < code_num:
            _a = random.randint(0, 35)
            str = str + ver_dict[_a]
            _nq = _nq + 1
        _tmp_list.append(str)
    for i in _tmp_list:
        image = ImageCaptcha()
        captcha = image.generate(i)
        captcha_image = Image.open(captcha)
        captcha_image.save(os.path.join(Paths_ts, f"{i}.jpg"))
        # captcha_image.show()


if __name__ == '__main__':
    _a = []
    while len(_a) < 10:
        create_test(256)
        _a.append(test_model())
    max_a = max(_a)
    min_a = min(_a)
    for i in _a:
        if max_a == i:
            _a.remove(i)
        if min_a == i:
            _a.remove(i)
    id = 0
    for ic in _a:
        id = id + ic
    print(id / len(_a))
