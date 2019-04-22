import os

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy
from torch.utils import data

from tqdm import tqdm
import click
import numpy as np

from pspnet import PSPNet
from PIL import Image
#import cv2

import glob

device = torch.device("cuda")

current = os.getcwd()

class Dataset(data.Dataset):
    def __init__(self, datapath, maskpath, classes):
        self.all_data = glob.glob(datapath +'/*.jpg')
        self.all_mask = glob.glob(maskpath +'/*.png')
        self.classes = classes

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trans_mask = transforms.Compose([
        transforms.Resize((224,224))])
        #transforms.Resize(224)])
        
        img = Image.open(self.all_data[index])
        img = img.convert('RGB')
        img = trans(img)
        img = img.type(torch.FloatTensor)
        mask = Image.open(self.all_mask[index]).convert('L')
        mask = trans_mask(mask)
        y_cls = np.zeros(self.classes)
        y_cls[np.unique(mask)] = 1
        mask = transforms.ToTensor()(mask)
        mask = mask.view(224,224)
        mask = mask.type(torch.LongTensor)
        y_cls = torch.from_numpy(y_cls)
        y_cls = y_cls.type(torch.FloatTensor)
        
        
        
        return (img,mask,y_cls)




models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    #net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    #net = net.cuda()
    return net, epoch


@click.command()
@click.option('--backend', type=str, default='resnet18', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=16)
@click.option('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.001)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')






def predict(backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, milestones, gpu):
    image_name = '/mountain.jpg'
    net, starting_epoch = build_network(snapshot, backend)
    

    net.load_state_dict(torch.load(current + '/models_path/PSPNet_4'))
    net.eval()

    net = nn.DataParallel(net)
    net.to(device)

    image_path = current + image_name
    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = trans(img)
    img = img.type(torch.FloatTensor)
    img = img.to(device)
    img.unsqueeze_(0)
    out,out_cls = net(img)
    out = torch.max(out,1)[1]
    print(out.shape)
    out = out.cpu().numpy()
    print(np.unique(out))
    plt.imshow(out.reshape(224,224))  
    plt.show()                                

       
if __name__ == '__main__':
    predict()
