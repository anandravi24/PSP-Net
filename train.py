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
@click.option('--data-path', type=str, help='Path to dataset folder')
@click.option('--models-path', type=str, help='Path for storing model snapshots')
@click.option('--backend', type=str, default='resnet50', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=16)
@click.option('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.001)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')



def train(data_path, models_path, backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, milestones, gpu):
   # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = build_network(snapshot, backend)
    net = nn.DataParallel(net)
    net.to(device)
    data_path = current +'/segmentation_data/images/training'
    mask_path = current + '/segmentation_data/annotations/training'
    models_path = current + '/models_path'
    
    validation_path = current +'/segmentation_dummy_data/images/validation'
    validation_mask = current + '/segmentation_dummy_data/annotations/validation'
    
    
    #os.makedirs(models_path, exist_ok=True)

    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
 
   '''



    training_dataset = Dataset(data_path, mask_path, classes = 112) 	
    validation_dataset = Dataset(validation_path, validation_mask, classes = 112)
    
    
    training_loader = data.DataLoader(training_dataset, batch_size = 64, num_workers = 32)
    validation_loader = data.DataLoader(validation_dataset)
    
    
    max_steps = 2
    train_loader, class_weights, n_images = None, None, None
    
    optimizer = optim.Adam(net.parameters(), lr=start_lr)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])
    steps = 0
    #val_loss = torch.Tensor([0])
    val_all = []
    for epoch in range(starting_epoch, starting_epoch + epochs):
        seg_criterion = nn.NLLLoss2d(weight=class_weights)
        cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        epoch_losses = []
        
        train_iterator = tqdm(training_loader, total=max_steps // batch_size + 1)
        net.train()
        for x, y, y_cls in train_iterator:
            steps += batch_size
            optimizer.zero_grad()
            x, y, y_cls = x.to(device), y.to(device), y_cls.to(device)
            out, out_cls = net(x)
            seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
            loss = seg_loss + alpha * cls_loss
            epoch_losses.append(loss.item())
            #status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, LR = {3:0.7f}'.format(epoch + 1, loss.data[0], np.mean(epoch_losses), scheduler.get_lr()[0])
            status = '[{0}] loss = {1:0.5f} val = {2:0.5f}'.format(epoch + 1, loss.item(), np.mean(val_all))
            train_iterator.set_description(status)
            loss.backward()
            optimizer.step()
        scheduler.step()
        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1)])))
        train_loss = np.mean(epoch_losses)
        validation_iterator = tqdm(validation_loader)
        for x,y,y_cls in validation_iterator:
            val_all = []
            x,y,y_cls = x.to(device), y.to(device), y_cls.to(device)
            out,out_cls = net(x)
            seg_loss, cls_loss = seg_criterion(out,y), cls_criterion(out_cls,y_cls)
            val_loss = seg_loss + alpha *cls_loss
            val_all.append(val_loss.item())

    #img = Image.open('mountain.jpg')
    #img = trans(img)
    #print(img.shape)
    #prediction = net(img.view(1,3,224,224))
    #print(prediction[0].shape)
    #print(torch.argmax(prediction[0], dim=1).shape, prediction[1].shape)
    #plt.plot(prediction[0].view(224, 224).numpy())  
    #plt.show()                                

       
if __name__ == '__main__':
    train()
