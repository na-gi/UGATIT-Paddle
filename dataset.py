# import torch.utils.data as data

from PIL import Image

import os
import os.path
import numpy as np
from collections import OrderedDict
import cv2
import random

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images


def horizontal_flip(im):
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im


def normalize(im, mean, std):
    im = im / 255.0
    im -= mean
    im /= std
    return im
    

def transform(image, info):
    image = np.asarray(image)

    # RandomHorizontalFlip
    if 'random_horizontalflip_prob' in info:
        transform_RandomHorizontalFlip = RandomHorizontalFlip(info['random_horizontalflip_prob'])
        image = transform_RandomHorizontalFlip(image)

    # Resize
    if 'resize_size' in info:
        transform_Resize = Resize(info['resize_size'])
        image = transform_Resize(image)

    # RandomPaddingCrop
    if 'crop_size' in info:
        transform_RandomPaddingCrop = RandomPaddingCrop(info['crop_size'])
        image = transform_RandomPaddingCrop(image)
    
    Normalize
    normalize = Normalize()
    image = normalize(image)

    # return Image.fromarray(np.uint8(image))   # 输出格式为PIL
    return image                                # 输出格式为narray


class DatasetFolder(object):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = transform(sample, self.transform)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

""" 给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象 """
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples


def myreader(data):
    def r():
        for i in range(len(data)):
            yield data[i][0].transpose(2,0,1).reshape(1, 3, 256, 256).astype('float32')
    return r

""" Transform """
class Resize:
    """调整图像大小（resize），当存在标注图像时，则同步进行处理。
    - 当目标大小（target_size）类型为int时，根据插值方式，
      将图像resize为[target_size, target_size]。
    - 当目标大小（target_size）类型为list或tuple时，根据插值方式，
      将图像resize为target_size, target_size的输入应为[w, h]或（w, h）。
    Args:
        target_size (int|list|tuple): 目标大小。
        interp (str): resize的插值方式，与opencv的插值方式对应，
            可选的值为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']，默认为"LINEAR"。
    Raises:
        TypeError: target_size不是int/list/tuple。
        ValueError:  target_size为list/tuple时元素个数不等于2。
        AssertionError: interp的取值不在['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']之内。
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size, interp='LINEAR'):
        self.interp = interp
        assert interp in self.interp_dict, "interp should be one of {}".format(
            interp_dict.keys())
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'
                    .format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。
        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
                其中，im_info跟新字段为：
                    -shape_before_resize (tuple): 保存resize之前图像的形状(h, w）。
        Raises:
            ZeroDivisionError: im的短边为0。
            TypeError: im不是np.ndarray数据。
            ValueError: im不是3维nd.ndarray。
        """
        if im_info is None:
            im_info = OrderedDict()
        im_info['resize'] = im.shape[:2]

        if not isinstance(im, np.ndarray):
            raise TypeError("ResizeImage: image type is not np.ndarray.")
        if len(im.shape) != 3:
            raise ValueError('ResizeImage: image is not 3-dimensional.')
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if float(im_size_min) == 0:
            raise ZeroDivisionError('ResizeImage: min size of image is 0')

        if isinstance(self.target_size, int):
            resize_w = self.target_size
            resize_h = self.target_size
        else:
            resize_w = self.target_size[0]
            resize_h = self.target_size[1]
        im_scale_x = float(resize_w) / float(im_shape[1])
        im_scale_y = float(resize_h) / float(im_shape[0])

        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp_dict[self.interp])
        if label is not None:
            label = cv2.resize(
                label,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp_dict['NEAREST'])
        if label is None:
            return im
        else:
            return (im, im_info, label)


class RandomHorizontalFlip:
    """以一定的概率对图像进行水平翻转。当存在标注图像时，则同步进行翻转。
    Args:
        prob (float): 随机水平翻转的概率。默认值为0.5。
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。
        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if random.random() < self.prob:
            im = horizontal_flip(im)
            if label is not None:
                label = horizontal_flip(label)
        if label is None:
            return im
        else:
            return (im, im_info, label)


class RandomPaddingCrop:
    """对图像和标注图进行随机裁剪，当所需要的裁剪尺寸大于原图时，则进行padding操作。
    Args:
        crop_size (int|list|tuple): 裁剪图像大小。默认为512。
        im_padding_value (list): 图像padding的值。默认为[127.5, 127.5, 127.5]。
        label_padding_value (int): 标注图像padding的值。默认值为255。
    Raises:
        TypeError: crop_size不是int/list/tuple。
        ValueError:  target_size为list/tuple时元素个数不等于2。
    """

    def __init__(self,
                 crop_size=512,
                 im_padding_value=[127.5, 127.5, 127.5],
                 label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError(
                    'when crop_size is list or tuple, it should include 2 elements, but it is {}'
                    .format(crop_size))
        elif not isinstance(crop_size, int):
            raise TypeError(
                "Type of crop_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(crop_size)))
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。
         Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        img_height = im.shape[0]
        img_width = im.shape[1]

        if img_height == crop_height and img_width == crop_width:
            if label is None:
                return im
            else:
                return (im, im_info, label)
        else:
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):
                im = cv2.copyMakeBorder(
                    im,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.im_padding_value)
                if label is not None:
                    label = cv2.copyMakeBorder(
                        label,
                        0,
                        pad_height,
                        0,
                        pad_width,
                        cv2.BORDER_CONSTANT,
                        value=self.label_padding_value)
                img_height = im.shape[0]
                img_width = im.shape[1]

            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)

                im = im[h_off:(crop_height + h_off), w_off:(w_off + crop_width
                                                            ), :]
                if label is not None:
                    label = label[h_off:(crop_height + h_off), w_off:(
                        w_off + crop_width)]
        if label is None:
            return im
        else:
            return (im, im_info, label)



class Normalize():
    """对图像进行标准化。
    1.尺度缩放到 [0,1]。
    2.对图像进行减均值除以标准差操作。
    Args:
        mean (list): 图像数据集的均值。默认值[0.5, 0.5, 0.5]。
        std (list): 图像数据集的标准差。默认值[0.5, 0.5, 0.5]。
    Raises:
        ValueError: mean或std不是list对象。std包含0。
    """

    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, list) and isinstance(self.std, list)):
            raise ValueError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。
         Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)

        if label is None:
            return im
        else:
            return (im, im_info, label)