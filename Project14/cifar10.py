import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg16
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img



from autoaugment import CIFAR10Policy
from PIL import Image, ImageEnhance, ImageOps
import PIL

'''
transform=transforms.Compose(
                        [transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
                        transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
			            transforms.ToTensor(), 
                        Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

trainset = datasets.CIFAR10(root='./data', train=True,
                            transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=4)
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(i)
#print(trainloader.ToTensor())

image = PIL.Image.open('/Users/kartheek/Pictures/tony.jpg')
policy = CIFAR10Policy()
transformed = policy(image)
print(transformed.shape);

#data = ImageFolder(rootdir, )
#loader = DataLoader(data, ...)
'''
def show_sixteen(images, titles=0):
    f, axarr = plt.subplots(4, 4, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx])
        ax.axis("off")
        if titles: ax.set_title(titles[idx])
    plt.show()

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

'''
data = ImageFolder("data/cifar_data", transform=transforms.Compose([
    CIFAR10Policy(), 
    transforms.ToTensor(),
]))
loader = DataLoader(data, batch_size=1)
'''



#Autoaugment

transform_list= []
transform_list.append(transforms.Compose([
                        transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
                        transforms.ToTensor(),
                        ]))
transform_list.append(transforms.Compose([
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.ToTensor(),
                        ]))
transform_list.append(transforms.Compose([
                        Cutout(n_holes=1, length=16),
                        transforms.ToTensor(),
                        ]))
transform_list.append(transforms.Compose([
                        CIFAR10Policy(),
                        transforms.ToTensor(),
                        ]))
print(len(transform_list))
print("bla bvla")

for i in range(1):
    k = i%4
    i=100
    tmp = transform_list[k]
    '''
    if (i<3):
        tmp = transform_list[i]
    else:
        tmp = transform_list[3]
    '''

    transform_cut=transforms.Compose([
                        transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
                        transforms.RandomHorizontalFlip(p=1),
                        CIFAR10Policy(), 
                        transforms.ToTensor(), 
                        Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])
  

    trainset_cut = datasets.CIFAR10(root='./data', train=True,#download=True,
                            transform=transform_cut)


    trainloader_cut = torch.utils.data.DataLoader(trainset_cut, batch_size=1,
                                          shuffle=True, num_workers=4
                                          )  
    imgs, count = [], 0
    imgarr = []
    lablarr=[]



    for _ in range(1):
        for img1 in trainloader_cut:
        #print(img1[1].numpy()[0])
            ele = []
            lable = img1[1].numpy()[0]
            img1 = np.transpose(img1[0][0].numpy()*255, (1,2,0)).astype(np.uint8)
            
            #ele.append(img1)
            #ele.append(lable)
            imgarr.append(img1)
            lablarr.append(lable)
        #print(img1.shape)

        #imgs.append(img1)
        #imgs.append(img2)
        
    print(len(imgarr))
    img_arr = np.array(imgarr,ndmin=4)
    labl_arr = np.array(lablarr,ndmin=1)
    print(img_arr.shape)
    print(labl_arr.shape)
    np.save(str(i)+'_x',img_arr)
    np.save(str(i)+'_y',labl_arr)

    #np.save('aug_augi',img_arr)
'''
imgs = []
for i in range(16):
    imgs.append(img_arr[i])
    print(labl_arr[i])
#print(img_arr[0][0])
show_sixteen(imgs)
'''
'''
        #tmp = np.asarray(img_)
        x,y = img_
        print(type(img_))
        img_ = np.transpose(img_[0][0].numpy()*255, (1,2,0)).astype(np.uint8)
        print(type(img_),img_.shape)
        #for _ in range(16): imgs.append(policy(img_))
        #show_sixteen(imgs)
        #break
        #print(type(img_))
        
        imgs.append(img_)
        count += 1
        if count==16: 
            show_sixteen(imgs)
            imgs, count = [], 0
            break
        '''