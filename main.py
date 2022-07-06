import os
import random
import time
import torch
from PIL import Image
from torch import nn, functional
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from captcha.image import ImageCaptcha
import Pth_File

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
path_file = '/home/Data/AI_data/data_train'  # 训练集
pth_save = '/home/Data/AI_data/pth_5'  # 模型保存路径
path_test = "/home/Data/AI_data/data_test"  # 测试集
Pt = Pth_File.Pths(pth_save)


class Mydataset(Dataset):
    def __init__(self, file_path=path_file):
        super(Mydataset, self).__init__()
        self.path = file_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        name_l = os.listdir(self.path)
        img_path = os.path.join(self.path, name_l[item])
        img = Image.open(img_path)
        Img = self.transform(img)
        name = name_l[item]
        name = name.split('.')[0]
        label = create_label(name)
        return Img, label

    def __len__(self):
        return len(os.listdir(self.path))


def create_label(in_str="0000"):
    """
    生成损失函数的标签\n
    :param in_str: 字符串，任意数字和字母，注意传入来的必须是字符串
    :return: 返回cuda生成的标签，注意转换！！！！
    """
    tmep = []
    for i in in_str:
        for key, value in ver_dict.items():
            if value == i:
                tmep.append(key)
    dc = torch.tensor(tmep)
    return functional.F.one_hot(dc, 36).view(-1).double()


def Run():
    """
    训练模型入口
    :return:
    """
    model = models.efficientnet_b0(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 36 * 5)  # 这里是指需要识别验证码位数
    )
    model.cuda()
    model_run(model)


def create_test(in_num=100, Paths_ts=path_test):
    """
    随机生成4位验证码(数字+字母混合)
    :param in_num: 随机生成个数，默认生成100个验证码
    :param Paths_ts: 生成验证码保存的路径
    :return:
    """
    _tmp_ls = []
    while len(_tmp_ls) < in_num:
        _nq = 0
        str = ""
        while _nq < 5:
            _a = random.randint(0, 35)
            str = str + ver_dict[_a]
            _nq = _nq + 1
        _tmp_ls.append(str)
    for i in _tmp_ls:
        image = ImageCaptcha()
        captcha = image.generate(i)
        captcha_image = Image.open(captcha)
        captcha_image.save(os.path.join(Paths_ts, f"{i}.jpg"))


def load_model():
    """
    用于加载模型，继续训练
    :return:
    """
    min_path = Pt.BS_file(False)
    model = torch.load(min_path)
    model_run(model=model)


def test_model(paths):
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
    return acc


def model_run(model, epoch=10000000):
    """
    运行模块
    :param model:传入需要训练的模型或继续训练的模型
    :param epoch: 轮数，这里默认是10000
    :return:
    """
    dh = Mydataset()
    trains_loader = DataLoader(dataset=dh, pin_memory=True, shuffle=True, batch_size=108, num_workers=12)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss()
    _n = 0
    for ie in range(epoch):
        for _, data in enumerate(trains_loader):
            img = data[0].cuda()
            label = data[1].cuda()
            Img_trains = model(img)
            loss = loss_func(Img_trains, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ls = round(loss.item(), 6)
            print(f"loss:{ls},{ie + 1}-{_ + 1}")
            if (_ + 1) % 200 == 0:
                save_file = f"{pth_save}/vercode_{ie + 1}_{ls}.pth"
                Pt.Del_file()
                torch.save(model, save_file)
                create_test(200)
                acc = test_model(save_file)
                now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                if os.path.exists("/home/Data/AI_data/logs/ver.log"):
                    if _n == 0:
                        os.remove("/home/Data/AI_data/logs/ver.log")
                        _n = _n + 1
                with open("/home/Data/AI_data/logs/ver.log", "a+") as Fs:
                    Fs.write(f"{now_time}\t第{ie + 1}轮,{_ + 1}组,准确率是{acc}%\n")
                Fs.close()
                os.popen("rm /home/Data/AI_data/data_test/*")


def main():
    if len(os.listdir(pth_save)) > 0:
        print("模型存在,自动加载最近保存模型")
        load_model()
    else:
        print("模型不存在,重新训练")
        Run()


main()
