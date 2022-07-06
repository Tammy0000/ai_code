import os
import time


class Pths:
    def __init__(self, pth_path):
        self.pth_path = pth_path
        self.tmp_list = []
        self.time_dict = {}

    def File_time(self):
        self.tmp_list.clear()
        self.time_dict.clear()
        Now_time = time.time()
        for i in os.listdir(self.pth_path):
            string = os.path.join(self.pth_path, i)
            _time = os.stat(string).st_mtime
            id_time = round(Now_time - _time)
            self.time_dict[id_time] = string
            self.tmp_list.append(id_time)

    def Del_file(self, lit=10):
        """
        删除文件
        :param lit:需要保留文件个数
        """
        self.File_time()
        while len(self.tmp_list) > lit:
            _max = max(self.tmp_list)
            path = self.time_dict[_max]
            os.remove(path)
            self.File_time()

    def BS_file(self, bool_re=True):
        """
        返回当前文件夹修改时间最长或者最短时间的文件(最长时间，创建时间最长，反之则创建时间最短)
        :param bool_re: True返回最大时间值，False 返回最小时间值路径
        :return: 最大值或者最小值的路径
        """
        self.File_time()
        if bool_re:
            return self.time_dict[max(self.tmp_list)]
        else:
            return self.time_dict[min(self.tmp_list)]
