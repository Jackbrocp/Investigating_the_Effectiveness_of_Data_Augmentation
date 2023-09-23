import  random, torch
import torchvision.transforms as transforms
from . import augmentation


M = 10

color_range = torch.arange(0, 0.9+1e-8, (0.9-0)/M).tolist()
rotate_range = torch.arange(0, 30+1e-8, (30-0)/M).tolist()
shear_range = torch.arange(0, 0.3+1e-8, (0.3-0)/M).tolist()
translate_range = torch.arange(0, 10+1e-8, (10-0)/M).tolist()


Mag = {'Brightness' : color_range, 'Color' : color_range, 'Contrast' : color_range, 
       'Posterize' : torch.arange(4, 8+1e-8, (8-4)/M).tolist()[::-1], 'Sharpness' : color_range, 
       'Solarize' : torch.arange(0, 256+1e-8, (256-0)/M).tolist()[::-1], 
       
       'Cutout' : torch.arange(0, 32*0.15+1e-8, (32*0.15-0)/M),
       
       'Rotate' : rotate_range, 'ShearX' : shear_range, 'ShearY' : shear_range,
       'TranslateX' : translate_range, 'TranslateY' : translate_range}


Fun = {'AutoContrast' : augmentation.AutoContrast, 'Brightness' : augmentation.Brightness, 
       'Color' : augmentation.Color, 'Contrast' : augmentation.Contrast,  'Equalize' : augmentation.Equalize, 'Invert' : augmentation.Invert, 
       'Posterize' : augmentation.Posterize, 'Sharpness' : augmentation.Sharpness, 'Solarize' : augmentation.Solarize, 
       
       'Cutout' : augmentation.Cutout,
         
       'Rotate' : augmentation.Rotate, 'ShearX' : augmentation.ShearX, 'ShearY' : augmentation.ShearY, 
       'TranslateX' : augmentation.TranslateX, 'TranslateY' : augmentation.TranslateY}


class Policy(torch.nn.Module):
    def __init__(self, policy, pre_transform, post_transform):
        super().__init__()
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        
        if policy == 'cifar': self.policy = cifar()
        elif policy == 'cifar_code': self.policy = cifar_code()
        elif policy == 'svhn': self.policy = svhn()
        elif policy == 'imagenet': self.policy = imagenet()

    def forward(self, image):
        policy_idx = random.randint(0, len(self.policy)-1)
        if self.post_transform == None and self.pre_transform==None:
            policy_transform = self.policy[policy_idx]
        else:
            policy_transform = self.pre_transform + self.policy[policy_idx] + self.post_transform
        # else:
        #     policy_transform = self.pre_transform + self.policy[policy_idx]
        policy_transform = transforms.Compose(policy_transform)
        image = policy_transform(image)
        return image 
    
    
def SubPolicy(f1, p1, m1, f2, p2, m2):
    subpolicy = []
    if f1 in ['AutoContrast', 'Equalize', 'Invert']: subpolicy.append(Fun[f1](p1))
    else: subpolicy.append(Fun[f1](p1, Mag[f1][m1]))
    
    if f2 in ['AutoContrast', 'Equalize', 'Invert']: subpolicy.append(Fun[f2](p2))
    else: subpolicy.append(Fun[f2](p2, Mag[f2][m2]))
        
    return subpolicy
    

def cifar_code():
    exp0_0 = [SubPolicy('Invert', 0.1, 7,                    'Contrast', 0.2, 6), 
              SubPolicy('Rotate', 0.7, 2,                    'TranslateX', 0.3, 9),
              SubPolicy('Sharpness', 0.8, 1,                 'Sharpness', 0.9, 3),
              SubPolicy('ShearY', 0.5, 8,                    'TranslateY', 0.7, 9),
              SubPolicy('AutoContrast', 0.5, 8,              'Equalize', 0.9, 2)]
    exp0_1 = [SubPolicy('Solarize', 0.4, 5,                  'AutoContrast', 0.9, 3),
              SubPolicy('TranslateY', 0.9, 9,                'TranslateY', 0.7, 9),
              SubPolicy('AutoContrast', 0.9, 2,              'Solarize', 0.8, 3),
              SubPolicy('Equalize', 0.8, 8,                  'Invert', 0.1, 3),
              SubPolicy('TranslateY', 0.7, 9,                'AutoContrast', 0.9, 1)]
    exp0_2 = [SubPolicy('Solarize', 0.4, 5,                  'AutoContrast', 0.0, 2),
              SubPolicy('TranslateY', 0.7, 9,                'TranslateY', 0.7, 9),
              SubPolicy('AutoContrast', 0.9, 0,              'Solarize', 0.4, 3),
              SubPolicy('Equalize', 0.7, 5,                  'Invert', 0.1, 3),
              SubPolicy('TranslateY', 0.7, 9,                'TranslateY', 0.7, 9)]
    exp0_3 = [SubPolicy('Solarize', 0.4, 5,                  'AutoContrast', 0.9, 1),
              SubPolicy('TranslateY', 0.8, 9,                'TranslateY', 0.9, 9),
              SubPolicy('AutoContrast', 0.8, 0,              'TranslateY', 0.7, 9),
              SubPolicy('TranslateY', 0.2, 7,                'Color', 0.9, 6),
              SubPolicy('Equalize', 0.7, 6,                  'Color', 0.4, 9)]
    
    exp1_0 = [SubPolicy('ShearY', 0.2, 7,                    'Posterize', 0.3, 7),
              SubPolicy('Color', 0.4, 3,                     'Brightness', 0.6, 7),
              SubPolicy('Sharpness', 0.3, 9,                 'Brightness', 0.7, 9),
              SubPolicy('Equalize', 0.6, 5,                  'Equalize', 0.5, 1),
              SubPolicy('Contrast', 0.6, 7,                  'Sharpness', 0.6, 5)]
    exp1_1 = [SubPolicy('Brightness', 0.3, 7,                'AutoContrast', 0.5, 8),
              SubPolicy('AutoContrast', 0.9, 4,              'AutoContrast', 0.5, 6),
              SubPolicy('Solarize', 0.3, 5,                  'Equalize', 0.6, 5),
              SubPolicy('TranslateY', 0.2, 4,                'Sharpness', 0.3, 3),
              SubPolicy('Brightness', 0.0, 8,                'Color', 0.8, 8)]
    exp1_2 = [SubPolicy('Solarize', 0.2, 6,                  'Color', 0.8, 6),
              SubPolicy('Solarize', 0.2, 6,                  'AutoContrast', 0.8, 1),
              SubPolicy('Solarize', 0.4, 1,                  'Equalize', 0.6, 5),
              SubPolicy('Brightness', 0.0, 0,                'Solarize', 0.5, 2),
              SubPolicy('AutoContrast', 0.9, 5,              'Brightness', 0.5, 3)]
    exp1_3 = [SubPolicy('Contrast', 0.7, 5,                  'Brightness', 0.0, 2),
              SubPolicy('Solarize', 0.2, 8,                  'Solarize', 0.1, 5),
              SubPolicy('Contrast', 0.5, 1,                  'TranslateY', 0.2, 9),
              SubPolicy('AutoContrast', 0.6, 5,              'TranslateY', 0.0, 9),
              SubPolicy('AutoContrast', 0.9, 4,              'Equalize', 0.8, 4)]
    exp1_4 = [SubPolicy('Brightness', 0.0, 7,                'Equalize', 0.4, 7),
              SubPolicy('Solarize', 0.2, 5,                  'Equalize', 0.7, 5),
              SubPolicy('Equalize', 0.6, 8,                  'Color', 0.6, 2),
              SubPolicy('Color', 0.3, 7,                     'Color', 0.2, 4),
              SubPolicy('AutoContrast', 0.5, 2,              'Solarize', 0.7, 2)]
    exp1_5 = [SubPolicy('AutoContrast', 0.2, 0,              'Equalize', 0.1, 0),
              SubPolicy('ShearY', 0.6, 5,                    'Equalize', 0.6, 5),
              SubPolicy('Brightness', 0.9, 3,                'AutoContrast', 0.4, 1),
              SubPolicy('Equalize', 0.8, 8,                  'Equalize', 0.7, 7),
              SubPolicy('Equalize', 0.7, 7,                  'Solarize', 0.5, 0)]
    exp1_6 = [SubPolicy('Equalize', 0.8, 4,                  'TranslateY', 0.8, 9),
              SubPolicy('TranslateY', 0.8, 9,                'TranslateY', 0.6, 9),
              SubPolicy('TranslateY', 0.9, 0,                'TranslateY', 0.5, 9),
              SubPolicy('AutoContrast', 0.5, 3,              'Solarize', 0.3, 4),
              SubPolicy('Solarize', 0.5, 3,                  'Equalize', 0.4, 4)]
    
    exp2_0 = [SubPolicy('Color', 0.7, 7,                     'TranslateX', 0.5, 8),
              SubPolicy('Equalize', 0.3, 7,                  'AutoContrast', 0.4, 8),
              SubPolicy('TranslateY', 0.4, 3,                'Sharpness', 0.2, 6),
              SubPolicy('Brightness', 0.9, 6,                'Color', 0.2, 8),
              SubPolicy('Solarize', 0.5, 2,                  'Invert', 0.0, 3)]
    exp2_1 = [SubPolicy('AutoContrast', 0.1, 5,              'Brightness', 0.0, 0),
              SubPolicy('Cutout', 0.2, 4,                    'Equalize', 0.1, 1),
              SubPolicy('Equalize', 0.7, 7,                  'AutoContrast', 0.6, 4),
              SubPolicy('Color', 0.1, 8,                     'ShearY', 0.2, 3),
              SubPolicy('ShearY', 0.4, 2,                    'Rotate', 0.7, 0)]
    exp2_2 = [SubPolicy('ShearY', 0.1, 3,                    'AutoContrast', 0.9, 5),
              SubPolicy('TranslateY', 0.3, 6,                'Cutout', 0.3, 3),
              SubPolicy('Equalize', 0.5, 0,                  'Solarize', 0.6, 6),
              SubPolicy('AutoContrast', 0.3, 5,              'Rotate', 0.2, 7),
              SubPolicy('Equalize', 0.8, 2,                  'Invert', 0.4, 0)]
    exp2_3 = [SubPolicy('Equalize', 0.9, 5,                  'Color', 0.7, 0),
              SubPolicy('Equalize', 0.1, 1,                  'ShearY', 0.1, 3),
              SubPolicy('AutoContrast', 0.7, 3,              'Equalize', 0.7, 0),
              SubPolicy('Brightness', 0.5, 1,                'Contrast', 0.1, 7),
              SubPolicy('Contrast', 0.1, 4,                  'Solarize', 0.6, 5)]
    exp2_4 = [SubPolicy('Solarize', 0.2, 3,                  'ShearX', 0.0, 0),
              SubPolicy('TranslateX', 0.3, 0,                'TranslateX', 0.6, 0),
              SubPolicy('Equalize', 0.5, 9,                  'TranslateY', 0.6, 7),
              SubPolicy('ShearX', 0.1, 0,                    'Sharpness', 0.5, 1),
              SubPolicy('Equalize', 0.8, 6,                  'Invert', 0.3, 6)]
    exp2_5 = [SubPolicy('AutoContrast', 0.3, 9,              'Cutout', 0.5, 3),
              SubPolicy('ShearX', 0.4, 4,                    'AutoContrast', 0.9, 2),
              SubPolicy('ShearX', 0.0, 3,                    'Posterize', 0.0, 3),
              SubPolicy('Solarize', 0.4, 3,                  'Color', 0.2, 4),
              SubPolicy('Equalize', 0.1, 4,                  'Equalize', 0.7, 6)]
    exp2_6 = [SubPolicy('Equalize', 0.3, 8,                  'AutoContrast', 0.4, 3),
              SubPolicy('Solarize', 0.6, 4,                  'AutoContrast', 0.7, 6),
              SubPolicy('AutoContrast', 0.2, 9,              'Brightness', 0.4, 8),
              SubPolicy('Equalize', 0.1, 0,                  'Equalize', 0.0, 6),
              SubPolicy('Equalize', 0.8, 4,                  'Equalize', 0.0, 4)]
    exp2_7 = [SubPolicy('Equalize', 0.5, 5,                  'AutoContrast', 0.1, 2),
              SubPolicy('Solarize', 0.5, 5,                  'AutoContrast', 0.9, 5),
              SubPolicy('AutoContrast', 0.6, 1,              'AutoContrast', 0.7, 8),
              SubPolicy('Equalize', 0.2, 0,                  'AutoContrast', 0.1, 2),
              SubPolicy('Equalize', 0.6, 9,                  'Equalize', 0.4, 4)]
    
    exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3
    exp1s = exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
    exp2s = exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7
    return  exp0s + exp1s + exp2s
    
    
def cifar():
    policy = [SubPolicy('Invert', 0.1, 7,                    'Contrast', 0.2, 6),
              SubPolicy('Rotate', 0.7, 2,                    'TranslateX', 0.3, 9),
              SubPolicy('Sharpness', 0.8, 1,                 'Sharpness', 0.9, 3),
              SubPolicy('ShearY', 0.5, 8,                    'TranslateY', 0.7, 9),
              SubPolicy('AutoContrast', 0.5, 8,              'Equalize', 0.9, 2),
              SubPolicy('ShearY', 0.2, 7,                    'Posterize', 0.3, 7),
              SubPolicy('Color', 0.4, 3,                     'Brightness', 0.6, 7),
              SubPolicy('Sharpness', 0.3, 9,                 'Brightness', 0.7, 9),
              SubPolicy('Equalize', 0.6, 5,                  'Equalize', 0.5, 1),
              SubPolicy('Contrast', 0.6, 7,                  'Sharpness', 0.6, 5),
              SubPolicy('Color', 0.7, 7,                     'TranslateX', 0.5, 8),
              SubPolicy('Equalize', 0.3, 7,                  'AutoContrast', 0.4, 8),
              SubPolicy('TranslateY', 0.4, 3,                'Sharpness', 0.2, 6),
              SubPolicy('Brightness', 0.9, 6,                'Color', 0.2, 8),
              SubPolicy('Solarize', 0.5, 2,                  'Invert', 0.0, 3),
              SubPolicy('Equalize', 0.2, 0,                  'AutoContrast', 0.6, 0),
              SubPolicy('Equalize', 0.2, 8,                  'Equalize', 0.6, 4),
              SubPolicy('Color', 0.9, 9,                     'Equalize', 0.6, 6),
              SubPolicy('AutoContrast', 0.8, 4,              'Solarize', 0.2, 8),
              SubPolicy('Brightness', 0.1, 3,                'Color', 0.7, 0),
              SubPolicy('Solarize', 0.4, 5,                  'AutoContrast', 0.9, 3),
              SubPolicy('TranslateY', 0.9, 9,                'TranslateY', 0.7, 9),
              SubPolicy('AutoContrast', 0.9, 2,              'Solarize', 0.8, 3),
              SubPolicy('Equalize', 0.8, 8,                  'Invert', 0.1, 3),
              SubPolicy('TranslateY', 0.7, 9,                'AutoContrast', 0.9, 1)]
    return policy


def svhn():
    policy = [SubPolicy('ShearX', 0.9, 4,                    'Invert', 0.2, 3),
              SubPolicy('ShearY', 0.9, 8,                    'Invert', 0.7, 5),
              SubPolicy('Equalize', 0.6, 5,                  'Solarize', 0.6, 6),
              SubPolicy('Invert', 0.9, 3,                    'Equalize', 0.6, 3),
              SubPolicy('Equalize', 0.6, 1,                  'Rotate', 0.9, 3),
              SubPolicy('ShearX', 0.9, 4,                    'AutoContrast', 0.8, 3),
              SubPolicy('ShearY', 0.9, 8,                    'Invert', 0.4, 5),
              SubPolicy('ShearY', 0.9, 5,                    'Solarize', 0.2, 6),
              SubPolicy('Invert', 0.9, 6,                    'AutoContrast', 0.8, 1),
              SubPolicy('Equalize', 0.6, 3,                  'Rotate', 0.9, 3),
              SubPolicy('ShearX', 0.9, 4,                    'Solarize', 0.3, 3),
              SubPolicy('ShearY', 0.8, 8,                    'Invert', 0.7, 4),
              SubPolicy('Equalize', 0.9, 5,                  'TranslateY', 0.6, 6),
              SubPolicy('Invert', 0.9, 4,                    'Equalize', 0.6, 7),
              SubPolicy('Contrast', 0.3, 3,                  'Rotate', 0.8, 4),
              SubPolicy('Invert', 0.8, 5,                    'TranslateY', 0.0, 2),
              SubPolicy('ShearY', 0.7, 6,                    'Solarize', 0.4, 8),
              SubPolicy('Invert', 0.6, 4,                    'Rotate', 0.8, 4),
              SubPolicy('ShearY', 0.3, 7,                    'TranslateX', 0.9, 3),
              SubPolicy('ShearX', 0.1, 6,                    'Invert', 0.6, 5),
              SubPolicy('Solarize', 0.7, 2,                  'TranslateY', 0.6, 7),
              SubPolicy('ShearY', 0.8, 4,                    'Invert', 0.8, 8),
              SubPolicy('ShearX', 0.7, 9,                    'TranslateY', 0.8, 3),
              SubPolicy('ShearY', 0.8, 5,                    'AutoContrast', 0.7, 3),
              SubPolicy('ShearX', 0.7, 2,                    'TranslateY', 0.1, 5)]
    return policy

         
def imagenet():
    policy = [SubPolicy('Posterize', 0.4, 8,                 'Rotate', 0.6, 9),
              SubPolicy('Solarize', 0.6, 5,                  'AutoContrast', 0.6, 5),
              SubPolicy('Equalize', 0.8, 8,                  'Equalize', 0.6, 3),
              SubPolicy('Posterize', 0.6, 7,                 'Posterize', 0.6, 6),
              SubPolicy('Equalize', 0.4, 7,                  'Solarize', 0.2, 4),
              SubPolicy('Equalize', 0.4, 4,                  'Rotate', 0.8, 8),
              SubPolicy('Solarize', 0.6, 3,                  'Equalize', 0.6, 7),
              SubPolicy('Posterize', 0.8, 5,                 'Equalize', 1.0, 2),
              SubPolicy('Rotate', 0.2, 3,                    'Solarize', 0.6, 8),
              SubPolicy('Equalize', 0.6, 8,                  'Posterize', 0.4, 6),
              SubPolicy('Rotate', 0.8, 8,                    'Color', 0.4, 0),
              SubPolicy('Rotate', 0.4, 9,                    'Equalize', 0.6, 2),
              SubPolicy('Equalize', 0.0, 7,                  'Equalize', 0.8, 8),
              SubPolicy('Invert', 0.6, 4,                    'Equalize', 1.0, 8),
              SubPolicy('Color', 0.6, 4,                     'Contrast', 1.0, 8),
              SubPolicy('Rotate', 0.8, 8,                    'Color', 1.0, 2),
              SubPolicy('Color', 0.8, 8,                     'Solarize', 0.8, 7),
              SubPolicy('Sharpness', 0.4, 7,                 'Invert', 0.6, 8),
              SubPolicy('ShearX', 0.6, 5,                    'Equalize', 1.0, 9),
              SubPolicy('Color', 0.4, 0,                     'Equalize', 0.6, 3),
              SubPolicy('Equalize', 0.4, 7,                  'Solarize', 0.2, 4),
              SubPolicy('Solarize', 0.6, 5,                  'AutoContrast', 0.6, 5),
              SubPolicy('Invert', 0.6, 4,                    'Equalize', 1.0, 8),
              SubPolicy('Color', 0.6, 4,                     'Contrast', 1.0, 8),
              SubPolicy('Equalize', 0.8, 8,                  'Equalize', 0.6, 3)]
    return policy