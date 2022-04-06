import os
import re
import numpy as np
import cv2
import imageio
import numpy as np
import sys
import random

def read(file):
    if file.endswith('.float3'): return readFloat(file)
    elif file.endswith('.flo'): return readFlow(file)
    elif file.endswith('.ppm'): return readImage(file)
    elif file.endswith('.pgm'): return readImage(file)
    elif file.endswith('.png'): return readImage(file)
    elif file.endswith('.jpg'): return readImage(file)
    elif file.endswith('.pfm'): return readPFM(file)[0]
    else: raise Exception('don\'t know how to read %s' % file)

def write(file, data):
    if file.endswith('.float3'): return writeFloat(file, data)
    elif file.endswith('.flo'): return writeFlow(file, data)
    elif file.endswith('.ppm'): return writeImage(file, data)
    elif file.endswith('.pgm'): return writeImage(file, data)
    elif file.endswith('.png'): return writeImage(file, data)
    elif file.endswith('.jpg'): return writeImage(file, data)
    elif file.endswith('.pfm'): return writePFM(file, data)
    else: raise Exception('don\'t know how to write %s' % file)

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data

    return imageio.imread(name)

def writeImage(name, data):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return writePFM(name, data, 1)

    return imageio.imsave(name, data)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)

def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data

def writeFloat(name, data):
    f = open(name, 'wb')

    dim=len(data.shape)
    if dim>3:
        raise Exception('bad float file dimension: %d' % dim)

    f.write(('float\n').encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))

    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))

    data = data.astype(np.float32)
    if dim==2:
        data.tofile(f)

    else:
        np.transpose(data, (2, 0, 1)).tofile(f)


class DataLoaderKITTI(object):
    def __init__(self, left_path, right_path, gt_path, batch_size, patch_size=(256, 512), max_disp=128):
        self.left_path = left_path
        self.right_path = right_path
        self.gt_path = gt_path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_disp = max_disp

    def generator(self, train_size, is_training=True):
        left_data = os.listdir(self.left_path)
        right_data = os.listdir(self.right_path)
        labels = os.listdir(self.gt_path)
        left_data.sort(key=str.lower)
        right_data.sort(key=str.lower)
        labels.sort(key=str.lower)

        train_left = left_data[:train_size]
        train_right = right_data[:train_size]
        train_labels = labels[:train_size]

        val_left = left_data[train_size:]
        val_right = right_data[train_size:]
        val_labels = labels[train_size:]

        index = [i for i in range(train_size)]
        random.shuffle(index)
        shuffled_labels = []
        shuffled_left_data = []
        shuffled_right_data = []

        for i in index:
            shuffled_left_data.append(train_left[i])
            shuffled_right_data.append(train_right[i])
            shuffled_labels.append(train_labels[i])
        if is_training:
            for j in range(train_size // self.batch_size):
                left, right, label = self.load_batch(shuffled_left_data[j * self.batch_size: (j + 1) * self.batch_size],
                                                        shuffled_right_data[
                                                        j * self.batch_size: (j + 1) * self.batch_size],
                                                        shuffled_labels[j * self.batch_size: (j + 1) * self.batch_size],
                                                        is_training)
                left = np.array(left)
                right = np.array(right)
                label = np.array(label)
                yield left, right, label
        else:
            for j in range(40 // self.batch_size):
                left, right, label = self.load_batch(val_left[j * self.batch_size: (j + 1) * self.batch_size],
                                                        val_right[j * self.batch_size: (j + 1) * self.batch_size],
                                                        val_labels[j * self.batch_size: (j + 1) * self.batch_size],
                                                        is_training)
                left = np.array(left)
                right = np.array(right)
                label = np.array(label)
                yield left, right, label

    def load_batch(self, left, right, labels, is_training):
        batch_left = []
        batch_right = []
        batch_label = []
        for x, y, z in zip(left, right, labels):
            if is_training:
                crop_x = random.randint(0, 368 - self.patch_size[0])
                crop_y = random.randint(0, 1224 - self.patch_size[1])
            else:
                crop_x = (368 - self.patch_size[0]) // 2
                crop_y = (1224 - self.patch_size[1]) // 2

            x = cv2.imread(os.path.join(self.left_path, x))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x[crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1], :]
            x = self.mean_std(x)
            batch_left.append(x)

            y = cv2.imread(os.path.join(self.right_path, y))
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            y = y[crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1], :]
            y = self.mean_std(y)
            batch_right.append(y)

            # z = cv2.imread(self.gt_path + z)
            # z = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
            # z = z[crop_x: crop_x + self.patch_size[0], crop_y: crop_y + self.patch_size[1]]
            # z[z > (self.max_disp-1)] = self.max_disp - 1
            # batch_label.append(z)
            z=read(os.path.join(self.gt_path, z))
            batch_label.append(z)

        return batch_left, batch_right, batch_label

    @staticmethod
    def mean_std(inputs):
        inputs = np.float32(inputs) / 255.
        inputs[:, :, 0] -= 0.485
        inputs[:, :, 0] /= 0.229
        inputs[:, :, 1] -= 0.456
        inputs[:, :, 1] /= 0.224
        inputs[:, :, 2] -= 0.406
        inputs[:, :, 2] /= 0.225
        return inputs
