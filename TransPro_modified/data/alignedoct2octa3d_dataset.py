import os
from data.base_dataset3d import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchio as tio
import torch


class AlignedOCT2OCTA3DDataset(BaseDataset):
    """A dataset class for paired image dataset.

    OCT to OCTA 3D

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, phase, 'A')  # get the image directory
        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths

        self.dir_A = os.path.join(opt.dataroot, phase, 'A')  # get the image directory
        self.A_paths = []
        for file in os.listdir(self.dir_A):
            self.A_paths.append(file)
        self.A_paths = sorted(self.A_paths)  # get image paths

        # self.dir_B = os.path.join(opt.dataroot, phase, 'B')  # get the image directory
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get image paths

        self.dir_B = os.path.join(opt.dataroot, phase, 'B')  # get the image directory
        self.B_paths = []
        for file in os.listdir(self.dir_B):
            self.B_paths.append(file)
        self.B_paths = sorted(self.B_paths)  # get image paths

        print("AB paths", self.dir_A, self.dir_B)
        print("AB paths length", len(self.A_paths), len(self.B_paths))
        assert (len(self.A_paths) == len(self.B_paths))
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        # A_path = self.dir_A[index]
        # B_path = self.dir_B[index]

        # A = np.load(A_path, allow_pickle=True)

        # print ("A_paths is ", self.A_paths)
        # print ("directory is ", self.dir_A)
        import re
        def sort_key(filename):
            # for bmp original files
            numbers = re.findall(r'\d+', filename)
            return int(numbers[0]) if numbers else 0

        image_list = []
        sorted_images_A = sorted(os.listdir(os.path.join(self.dir_A, A_path)), key=sort_key)
        for i, img in enumerate(sorted_images_A):
            # if i < 100 or i > 304-100:
            #     continue
            image_array = np.array(Image.open(os.path.join(self.dir_A, A_path, img)))
            image_list.append(image_array)
        A = np.stack(image_list, axis=0)

        image_list = []
        sorted_images_B = sorted(os.listdir(os.path.join(self.dir_B, B_path)), key=sort_key)
        for i, img in enumerate(sorted_images_B):
            image_array = np.array(Image.open(os.path.join(self.dir_B, B_path, img)))
            image_list.append(image_array)
        B = np.stack(image_list, axis=0)

        # import matplotlib.pyplot as plt
        # A_proj = np.mean(A, axis=1).astype(np.uint8)
        # AA_proj = A_proj.squeeze()
        # plt.imshow(AA_proj, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # print("A shape is before convert", A.shape)
        # print(Image.open(A_path).size)
        # B = np.load(B_path, allow_pickle=True)
        A = np.expand_dims(A, axis=0)
        B = np.expand_dims(B, axis=0)

        # apply the same transform to both A and B

        transform_params = get_params(self.opt, A.shape)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        transform_list = []

        A = A_transform(A)
        B = B_transform(B)
        # print("A shape is after convert", A.shape)
        A = (A - A.min()) / (A.max() - A.min())
        B = (B - B.min()) / (B.max() - B.min())
        A = 2 * A - 1
        B = 2 * B - 1

        A = torch.from_numpy(A)
        B = torch.from_numpy(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        assert (len(self.A_paths) == len(self.B_paths))
        return len(self.A_paths)
