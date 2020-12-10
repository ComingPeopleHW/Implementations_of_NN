from torch.utils.data import Dataset
from torchvision.transforms import transforms as tf
import os
import glob
from PIL import Image
import torch


# custom dataset for DataLoader
class ImageDataset(Dataset):
    def __init__(self, root, transforms=None, istrain=True):
        self.transform = tf.Compose(transforms)
        self.istrain = istrain
        self.files_A = sorted(glob.glob(os.path.join(root, '%s' % 'trainA') + '/*.*'))
        # print(os.path.join(root, '%s' % 'trainA') + '/*.*')
        self.files_B = sorted(glob.glob(os.path.join(root, '%s' % 'trainB') + '/*.*'))

    def __getitem__(self, index):
        try:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
            # print(item_A.shape)
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        except:
            print('files_A:', self.files_A[index % len(self.files_A)])
            print('files_B:', self.files_B[index % len(self.files_B)])
        r, g, b = item_A[0] + 1, item_A[1] + 1, item_A[2] + 1  # after transform,the image pixel in in [-1,1],+1 to[0,2]
        gray_weight_A = 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 2.0  # Attention illumination map weight in [0,1]
        gray_weight_A = torch.unsqueeze(gray_weight_A, 0)  # convert to shape (c,h,w)
        # print(gray_weight_A.shape)
        # print(item_A.shape)
        return {'A': item_A, 'B': item_B, 'A_gray': gray_weight_A}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
