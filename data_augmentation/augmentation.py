"""
Coior Augmentation: Autocontrast, Brightness, Color, Contrast, Equalize, Invert, Posterize, Sharpness, Solarize
Mask Augmentation: Cutout
Geometric Augmentation: Rotate, ShearX, ShearY, TranslateX, TranslateY

"""

import math, torch, torchvision
from . import function
import torchvision.transforms.functional as F

from PIL import Image, ImageOps


### Coior Augmentation
class AutoContrast(torch.nn.Module):
    """
    Autocontrast the pixels of the given image.
    """
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, image):
        if torch.rand(1) < self.p:
            autocontrast_image = ImageOps.autocontrast(image)
            return autocontrast_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)
    
    
class Brightness(torch.nn.Module):
    """
    Adjust image brightness using magnitude.
    """
    def __init__(self, p, magnitude, minus=True):
        super().__init__()
        self.p = p
        self.magnitude = magnitude
        self.minus = minus

    def forward(self, image):
        if self.minus and (torch.rand(1) < 0.5): self.magnitude *= -1
        if torch.rand(1) < self.p:
            brightness_image = function.brightness(image, 1+self.magnitude)
            return brightness_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, magnitude={1})".format(self.p, self.magnitude)
        

 

class Color(torch.nn.Module):
    """
    Adjust image color balance using magnitude.
    """
    def __init__(self, p, magnitude, minus=True):
        super().__init__()
        self.p = p
        self.magnitude = magnitude
        self.minus = minus

    def forward(self, image):
        if self.minus and (torch.rand(1) < 0.5): self.magnitude *= -1
        if torch.rand(1) < self.p:
            color_image = function.color(image, 1+self.magnitude)
            return color_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, magnitude={1})".format(self.p, self.magnitude)
    
    
class Contrast(torch.nn.Module):
    """
    Adjust image contrast using magnitude.
    """
    def __init__(self, p, magnitude, minus=True):
        super().__init__()
        self.p = p
        self.magnitude = magnitude
        self.minus = minus

    def forward(self, image):
        if self.minus and (torch.rand(1) < 0.5): self.magnitude *= -1
        if torch.rand(1) < self.p:
            contrast_image = function.contrast(image, 1+self.magnitude)
            return contrast_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, magnitude={1})".format(self.p, self.magnitude)

        
class Equalize(torch.nn.Module):
    """
    Equalize the histogram of the given image.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, image):
        if torch.rand(1) < self.p:
            equalize_image = ImageOps.equalize(image)
            return equalize_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)
    
    
class Invert(torch.nn.Module):
    """
    Invert the given image.
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, image):
        if torch.rand(1) < self.p:
            invert_image = ImageOps.invert(image)
            return invert_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)
    

class Posterize(torch.nn.Module):
    """
    Posterize the image by reducing the number of bits for each color channel.
    """
    def __init__(self, p, bits):
        super().__init__()
        self.p = p
        self.bits = int(bits)

    def forward(self, image):
        if torch.rand(1) < self.p:
            posterize_image = ImageOps.posterize(image, self.bits)
            return posterize_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, bits={1})".format(self.p, self.bits)
    

class Sharpness(torch.nn.Module):
    """
    Adjust image sharpness using magnitude.
    """
    def __init__(self, p, magnitude, minus=True):
        super().__init__()
        self.p = p
        self.magnitude = magnitude
        self.minus = minus

    def forward(self, image):
        if self.minus and (torch.rand(1) < 0.5): self.magnitude *= -1
        if torch.rand(1) < self.p:
            sharpness_image = function.sharpness(image, 1+self.magnitude)
            return sharpness_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, magnitude={1})".format(self.p, self.magnitude)
    
    
class Solarize(torch.nn.Module):
    """
    Solarize the image by inverting all pixel values above a threshold.
    """
    def __init__(self, p, threshold):
        super().__init__()
        self.p = p
        self.threshold = int(threshold)

    def forward(self, image):
        if torch.rand(1) < self.p:
            solarize_image = ImageOps.solarize(image, self.threshold)
            return solarize_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, threshold={1})".format(self.p, self.threshold)
    

### Mask Augmentation
class Cutout(torch.nn.Module):
    """
    Apply cutout (https://arxiv.org/abs/1708.04552) to the image.
    This operation applies a (2*pad_size, 2*pad_size) mask of zeros to a random location within image.
    The pixel values filled in will be of the value replace.
    """
    def __init__(self, p, pad_size, replace=128):
        super().__init__()
        self.p = p
        self.pad_size = math.ceil(pad_size)
        self.replace = replace

    def forward(self, image):
        if torch.rand(1) < self.p:
            cutout_image = function.cutout(image, self.pad_size, self.replace)
            return cutout_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, pad_size={1})".format(self.p, self.pad_size)
    
    
### Geometric Augmentation
class Rotate(torch.nn.Module):
    """
    Rotate image by degrees.
    The pixel values filled in will be of the value replace.
    """
    def __init__(self, p, degrees, replace=128, minus=True):
        super().__init__()
        self.p = p
        self.degrees = degrees
        self.replace = replace
        self.minus = minus

    def forward(self, image):
        if self.minus and (torch.rand(1) < 0.5): self.degrees *= -1
        if torch.rand(1) < self.p:
            rotate_image = image.rotate(self.degrees, fillcolor=(self.replace, self.replace, self.replace))
            return rotate_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, degrees={1})".format(self.p, self.degrees)
        
        
class ShearX(torch.nn.Module):
    """
    Shear image on X-axis.
    The pixel values filled in will be of the value replace.
    Level is usually between -0.3~0.3.
    """
    def __init__(self, p, level, replace=128, minus=True):
        super().__init__()
        self.p = p
        self.level = level
        self.replace = replace
        self.minus = minus

    def forward(self, image):
        if self.minus and (torch.rand(1) < 0.5): self.level *= -1
        if torch.rand(1) < self.p:
            shear_image = image.transform(image.size, Image.AFFINE, (1, self.level, 0, 0, 1, 0), fillcolor=(self.replace, self.replace, self.replace))
            return shear_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, level={1})".format(self.p, self.level)
        
        
class ShearY(torch.nn.Module):
    """
    Shear image on Y-axis.
    The pixel values filled in will be of the value replace.
    Level is usually between -0.3~0.3.
    """
    def __init__(self, p, level, replace=128, minus=True):
        super().__init__()
        self.p = p
        self.level = level
        self.replace = replace
        self.minus = minus

    def forward(self, image):
        if self.minus and (torch.rand(1) < 0.5): self.level *= -1
        if torch.rand(1) < self.p:
            shear_image = image.transform(image.size, Image.AFFINE, (1, 0, 0, self.level, 1, 0), fillcolor=(self.replace, self.replace, self.replace))
            return shear_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, level={1})".format(self.p, self.level)
        
        
class TranslateX(torch.nn.Module):
    """
    Translate image on X-axis.
    The pixel values filled in will be of the value replace.
    """
    def __init__(self, p, pixels, replace=128, minus=True):
        super().__init__()
        self.p = p
        self.pixels = int(pixels)
        self.replace = replace
        self.minus = minus

    def forward(self, image):
        if self.minus and (torch.rand(1) < 0.5): self.pixels *= -1
        if torch.rand(1) < self.p:
            translate_image = image.transform(image.size, Image.AFFINE, (1, 0, -self.pixels, 0, 1, 0), fillcolor=(self.replace, self.replace, self.replace))
            return translate_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, pixels={1})".format(self.p, self.pixels)
    
    
class TranslateY(torch.nn.Module):
    """
    Translate image on Y-axis.
    The pixel values filled in will be of the value replace.
    """
    def __init__(self, p, pixels, replace=128, minus=True):
        super().__init__()
        self.p = p
        self.pixels = int(pixels)
        self.replace = replace
        self.minus = minus

    def forward(self, image):
        if self.minus and (torch.rand(1) < 0.5): self.pixels *= -1
        if torch.rand(1) < self.p:
            translate_image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, -self.pixels), fillcolor=(self.replace, self.replace, self.replace))
            return translate_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, pixels={1})".format(self.p, self.pixels)