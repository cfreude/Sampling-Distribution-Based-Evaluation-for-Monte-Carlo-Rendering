import os

import matplotlib.pyplot as plt
import numpy as np
import hashlib
import time
import io
import json
import pickle


def hash_file(_path):

    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    sha = hashlib.sha512()

    with open(_path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha.update(data)

    return sha.hexdigest()


def cache_to_disk(_path='./cache/'):

    if not os.path.exists(_path):
        print(('Creating cache dir %s' % _path))
        os.mkdir(_path)

    def decorator(_func):

        def helper(*_args):
            arg_dump = pickle.dumps(_args, -1)
            hash_hex = hashlib.sha256(arg_dump).hexdigest()
            cache_path = os.path.join(_path, '%s_%s.npy' % (_func.__name__, hash_hex))

            try:
                result = np.load(cache_path, allow_pickle=True)
                print(('Loaded cache %s' % cache_path))
            except IOError:
                print(('Cache not found %s' % cache_path))
                result = _func(*_args)
                np.save(cache_path, result, allow_pickle=True)

            return result

        return helper

    return decorator


def clip_percentile(_x, p=95):
    return np.clip(_x, 0, np.percentile(_x, p))


def luminance(rgb_img):
    dim = rgb_img.shape
    lf = [0.27, 0.67, 0.06]  # [0.2126, 0.7152, 0.0722]
    to_lum = np.tile(lf, [dim[0], dim[1], 1])
    return np.sum(to_lum * rgb_img, axis=2, dtype=np.float64)


def tone_map(hdr_image, method='reinhard', gamma=2.2):

    hdr_image_lum = luminance(hdr_image)
    delta = 0.001

    tmp = np.log(delta + hdr_image_lum)
    l_average = np.exp(np.mean(tmp))
    l_white = np.mean(hdr_image_lum[hdr_image_lum > np.quantile(hdr_image_lum, 0.999)])

    key_a = np.max([0.0, 1.5 - 1.5 / (l_average * 0.1 + 1.0)]) + 0.1

    scaled = (key_a / l_average) * hdr_image

    ldr_image = np.zeros(hdr_image.shape, dtype=np.float64)

    method_dict = {
        'log': 0,
        'linear': 1,
        'reinhard': 2,
        'reinhard_mod': 3,
        'naive': 4,
        'lum': 5,
        None: 6,
    }

    if method_dict[method] == 0:  # log
        ldr_image = np.log(hdr_image + 0.001)

    elif method_dict[method] == 1:  # linear
        max_lum = np.max(hdr_image_lum)
        ldr_image = hdr_image / max_lum

    elif method_dict[method] == 2:  # reinhard
        ldr_image = scaled / (1.0 + scaled)

    elif method_dict[method] == 3:  # reinhard_mod
        ldr_image = (scaled * (1.0 + scaled / np.power(l_white, 2.0))) / (1.0 + scaled)

    elif method_dict[method] == 4:  # naive
        ldr_image = hdr_image / (hdr_image + 1.0)

    elif method_dict[method] == 5:  # lum
        ldr_image = luminance(hdr_image)

    elif method_dict[method] == 6:  # none
        ldr_image = hdr_image

    if gamma != 1.0:
        ldr_image = np.power(ldr_image, 1.0/gamma)

    return ldr_image


class OnlineStats:

    # https://www.johndcook.com/blog/standard_deviation/

    def __init__(self, _dtype=float):
        self.n = 0
        self.oldM = 0
        self.newM = 0
        self.oldS = 0
        self.newS = 0

        self.dtype = _dtype
        self.minimum = None
        self.maximum = None

    def clear(self):
        self.__init__()

    def push(self, x):

        x = np.array(x, dtype=self.dtype)

        self.n += 1

        # See Knuth TAOCP vol 2, 3rd edition, page 232

        if self.n == 1:
            self.oldM = self.newM = x
            self.oldS = np.zeros(x.shape, dtype=self.dtype)
            self.minimum = self.maximum = x
        else:
            self.newM = self.oldM + (x - self.oldM) / self.n
            self.newS = self.oldS + (x - self.oldM) * (x - self.newM)
            self.minimum = np.minimum(self.minimum, x)
            self.maximum = np.maximum(self.maximum, x)

        # set up for next iteration
        self.oldM = self.newM
        self.oldS = self.newS

    def num_data_values(self):
        return self.n

    def mean(self):
        if self.n > 0:
            return self.newM
        else:
            try:
                return np.zeros(self.newM.shape, self.dtype)
            except AttributeError:
                return 0

    def variance(self, doff=1):

        if self.n > 1:
            return self.newS / float(self.n - doff)
        else:
            try:
                return np.zeros(self.newM.shape, self.dtype)
            except AttributeError:
                return 0

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def max(self):
        return self.maximum

    def min(self):
        return self.minimum


def get_time_str(_duration):
    unit_str = 'ms'
    if _duration >= 0.1:
        unit_str = 'sec.'
        if _duration >= 60.0:
            unit_str = 'min'
            _duration /= 60.0
    return '%.1f %s' % (_duration, unit_str)


class Timer:

    def __init__(self):
        self.start_time = None
        self.total_duration = 0

    def start(self):
        self.start_time = time.time()

    def stop(self, _tag=None):
        return 'Elapsed: %s' % get_time_str(self.total_duration)

    def eta(self, _i, _count):
        elapsed = time.time() - self.start_time
        avg = elapsed / float(_i)
        rem_time = avg * (_count - _i)
        return 'ETA: %s' % get_time_str(rem_time)


def write_dict_to_json(path, dict):

    with io.open(path, 'w', encoding='utf-8') as f:
        d = json.dumps(dict, ensure_ascii=False)
        f.write(str(d))


def read_dict_from_json(path):

    with io.open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def upsample_rbg(smaller_img, _factor=2):
    return smaller_img.repeat(_factor, axis=0).repeat(_factor, axis=1)


if __name__ == '__main__':
    x = np.array([[[1,0,0], [0,1,0]], [[0,0,1], [1,1,1]]])*255
    plt.imshow(x)
    plt.show()
    x = upsample_rbg(x, 3)
    plt.imshow(x)
    plt.show()
    print(x)
