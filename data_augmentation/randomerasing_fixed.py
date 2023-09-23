import torch
import numpy as np
import random
from PIL import Image 
import torchvision
import torchvision.transforms as transforms
class RandomErasing_fixed(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self,  length, p=1.0 ):
        print('===============',length)
        self.n_holes = 1
        self.length = length
        self.p = p
       
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if random.uniform(0, 1) > self.p:
            return img
        h = img.size(1)
        w = img.size(2)
        mean=[0.4914, 0.4822, 0.4465]

        y = np.random.randint(h)
        x = np.random.randint(w)
        if img.size(0) ==3:
            img[0, x:min(x+self.length, w), y:min(y+self.length,h)] =  mean[0]
            img[1, x:min(x+self.length, w), y:min(y+self.length,h)] =  mean[1]
            img[2, x:min(x+self.length, w), y:min(y+self.length,h)] =  mean[2]
        else:
            img[0, x:min(x+self.length, w), y:min(y+self.length,h)] =  mean[0]
        return img

        
if __name__ == '__main__':
    b = RandomErasing_fixed(8)
    img = Image.open('test.jpg')
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    img = transform(img)
    res = b(img)
    mean = torch.tensor((0.485, 0.456, 0.406))
    std = torch.tensor((0.229, 0.224, 0.225))
    MEAN = [-mean/std for mean, std in zip(mean, std)]
    STD = [1/std for std in std]
    denormalizer = transforms.Normalize(mean=MEAN, std=STD)
    res = denormalizer(res)
    res = transforms.ToPILImage()(res).convert('RGB')
    res.save('res.jpg')