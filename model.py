import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from ResNetSE import ResNetSE34
from video_encoder import Generator,VideoEncoder,PatchDiscriminator
from video_loader import VoxcelebDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from content_loss import LossCnt
import argparse

class FinalModel(nn.Module):
    def __init__(self, size,style_dim):
        super(FinalModel,self).__init__()

        self.a_encoder = ResNetSE34(nOut=style_dim)
        self.v_encoder = VideoEncoder(size)
        
        self.generator = Generator(size,style_dim=style_dim,n_mlp=8)


    def forward(self,image,audio):

        style = self.a_encoder(audio)
        image_latent = self.v_encoder(image)
        gen_images,_ = self.generator([style],image_latent)
      
        return gen_images,image_latent

def write_video(images,path,frame_rate=25,size=(256,256)):
    fourcc = cv2.VideoWriter_fourcc(*'PIM1')
    vOut = cv2.VideoWriter(path, fourcc, frame_rate, size)
    images = np.minimum(np.maximum((images +1) * 127.5,0),255).astype(np.uint8)
    if not os.path.isdir('./images'):
        os.makedirs('./images')
    i =0 
    for image in images:
        vOut.write(image)
        plt.imsave('./images/%d.png'%(i), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        i+=1
    vOut.release()
   
    pass

def my_collate(batch):
    source_video = np.array([item[0] for item in batch])
    source_audio = np.array([item[1] for item in batch])
    target_video = np.array([item[2] for item in batch])
    target_audio = np.array([item[3] for item in batch])

    return [torch.from_numpy(source_video), torch.from_numpy(source_audio),torch.from_numpy(target_video),torch.from_numpy(target_audio)]


def l1_loss(target,pred):
    return torch.mean(torch.abs(target - pred))


def save_model(opt,model,path,epoch):
    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save({
            'model_state_dict': model.state_dict(),
            'Opt_state_dict': opt.state_dict(),
            
            }, path + '/model_%d.pth'%(epoch))

    print('Model is saved at %s'%(path))
    pass

def load_model(path,model,optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['Opt_state_dict'])
    return model,optimizer

    
def train(size=128,style_dim=32,batchsize=2, epoch= 1000, n_worker=2, max_len=18,opt='adam', device = torch.device("cuda:0"),load_pretrained=False,model_path='./save_model'):
    ## load nodel
    model = FinalModel(size,style_dim)
    model.to(device)
    print('Loading model to device .....')


    ## load dataset
    data = VoxcelebDataset(resl = (size,size), max_len = max_len,step=4)

    dataloader = DataLoader(data, shuffle=True, batch_size=batchsize,collate_fn=my_collate,drop_last=False,num_workers=n_worker)

    ## optimizer
    if opt =='adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001) 
        print('Using Adam optimzer')
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        print('Using SGD optimzer')
    
    ## writer
    writer = SummaryWriter('./log')

    if load_pretrained:
        files = glob.glob(model_path + '/*')
        files.sort()
        if len(files)>0:
            model,optimizer = load_model(files[-1],model,optimizer)
            print('load model at %s'%(files[-1]))
        else:
            print("there are no pretrained model")
            print('Model will be trained from beginning')




    n_iter = 0
    for i in range(epoch):
        model.train()
        for source_video,source_audio,target_video,target_audio in dataloader:
            source_video = source_video.to(device).view(batchsize*(max_len -2),3,size,size)
            target_audio = target_audio.to(device).view(batchsize*(max_len -2),-1)
            target_video = target_video.to(device).view(batchsize*(max_len -2),3,size,size)

            gen_images = model(source_video,target_audio)

            loss = l1_loss(target_video,gen_images)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter+= 1

            writer.add_scalar('loss', loss, n_iter)

            print('Epoch : %d ;Iter : %d ; Loss : %0.4f '%(i, n_iter, loss))

        
        if not os.path.isdir('./video'):
            os.mkdir('./video')
        write_video(gen_images.cpu().detach().numpy().swapaxes(1,2).swapaxes(2,3),'./video/gen_%s.avi'%(n_iter),size=(size,size))            

        save_model(optimizer,model,'./save_model',i)
        


if __name__ == '__main__':
    train(load_pretrained=False)




    