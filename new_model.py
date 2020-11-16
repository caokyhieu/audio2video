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
from torch.autograd import grad
from custom_layer import TimeDistributed

class FinalModel(nn.Module):
    def __init__(self, size,style_dim,device):
        super(FinalModel,self).__init__()
        self.device =device
        self.a_encoder = ResNetSE34(nOut=style_dim)
        self.v_encoder = VideoEncoder(size)
        
        self.generator = Generator(size,style_dim=style_dim,n_mlp=8)
        self.discriminator = PatchDiscriminator(size)


    def interpolate(self,x,y):
        alpha = torch.rand((x.shape[0],1,1,1)).to(self.device)
        return alpha * x + (1-alpha) * y


    def compute_gradient_penalty(self,D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        # Get random interpolation between real and fake samples
        interpolates = self.interpolate(real_samples,fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
#         fake = nn.Parameter(nn.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self,image,audio):

        style = self.a_encoder(audio)
        image_latent = self.v_encoder(image)
        gen_images,_ = self.generator([style],image_latent)
      
        return gen_images,image_latent

    def forward_generator(self,image,audio):
        gen_images,image_latent = self.forward(image,audio)
        gen_pred = self.discriminator(gen_images)
        return gen_images,gen_pred


    def forward_discriminator(self,source_image,target_audio,target_image):
        gen_image,image_latent = self.forward(source_image,target_audio)
        gen_pred = self.discriminator(gen_image)
        target_pred = self.discriminator(target_image)
        gradients_img = self.compute_gradient_penalty(self.discriminator,target_image,gen_image)
        return gen_image,gen_pred,target_pred,gradients_img




def write_video(images,path,frame_rate=25,size=(256,256),name=''):
    fourcc = cv2.VideoWriter_fourcc(*'PIM1')
    vOut = cv2.VideoWriter(path, fourcc, frame_rate, size)
    images = np.minimum(np.maximum((images +1) * 127.5,0),255).astype(np.uint8)
    if not os.path.isdir('./images'):
        os.makedirs('./images')
    i =0 
    for image in images:
        vOut.write(image)
        plt.imsave('./images/%s_%d.png'%(name,i), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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

def w_loss(y_pred,y_true):
    return torch.mean((-y_true) * y_pred)

def save_model(opt,model,path,epoch):
    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save({
            'model_state_dict': model.state_dict(),
            'Opt_state_dict': opt.state_dict(),
            
            }, path + '/model_%d.pth'%(epoch))

    print('Model is saved at %s'%(path))
    pass

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def load_model(path,model,optimizer,warm_up=False):
    checkpoint = torch.load(path)
    if warm_up:
        model_dict = model.state_dict()
        state_dict = checkpoint['model_state_dict']
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # optimizer.load_state_dict(checkpoint['Opt_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['Opt_state_dict'])

    return model,optimizer

    
def train(size=128,style_dim=256,batchsize=1, epoch= 100000, n_worker=0, max_len=76,n_critic=5,opt='adam', half_window=1200,device = torch.device("cuda:0"),load_pretrained=False,model_path='./save_model/model_1.pth',warm_up=False):
    ## load nodel
    model = FinalModel(size,style_dim,device)
    model.to(device)
    print('Loading model to device .....')


    ## load dataset
    data = VoxcelebDataset(resl = (size,size), max_len = max_len,step=3,half_window=half_window)

    dataloader = DataLoader(data, shuffle=True, batch_size=batchsize,collate_fn=my_collate,drop_last=True,num_workers=n_worker)

    ## optimizer
    if opt =='adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001) 
        print('Using Adam optimzer')
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        print('Using SGD optimzer')
    
    ## writer
    writer = SummaryWriter('./log')

    if load_pretrained:
      
       
        model,optimizer = load_model(model_path,model,optimizer,warm_up)
        print('load model at %s'%(model_path))
    

    ### loss func 
    loss_func = LossCnt('/vinai/hieuck/generate_video_from_audio/Pytorch_VGGFACE_IR.py', '/vinai/hieuck/generate_video_from_audio/Pytorch_VGGFACE.pth', device)
    # l1_latent_loss = nn.L1Loss()
    # for g in optimizer.param_groups:
    #     g['lr'] = 0.001
    ## Add schedule for optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    n_iter = 0
    for i in range(epoch):
        model.train()
        for source_video,source_audio,target_video,target_audio in dataloader:
        
            source_video = source_video.to(device).view(-1,3,size,size)
            # batch = list(source_video.size())[0]
            print(target_audio.size())
            target_audio = target_audio.to(device).view(-1,half_window*2)
            target_video = target_video.to(device).view(-1,3,size,size)

            # gen_images,image_latent = model(source_video,target_audio)
            # gen_images_up = F.upsample(gen_images, size=(224, 224), mode= 'bilinear')
            # target_video_up = F.upsample(target_video, size=(224, 224), mode= 'bilinear')
            # target_image_latent = model.v_encoder(target_video)

            # gen_images,gen_pred,target_pred,gradients_img = model.forward_discriminator(source_video,target_audio,target_video)
            # gen_images_up = F.upsample(gen_images, size=(224, 224), mode= 'bilinear')
            # target_video_up = F.upsample(target_video, size=(224, 224), mode= 'bilinear')

            # d_loss =   5 * w_loss(gen_pred,- torch.ones(gen_pred.size()).to(device)) +  5 * w_loss(target_pred,torch.ones(target_pred.size()).to(device)) + 1e-3 * gradients_img
            
            # model.zero_grad()
            # d_loss.backward()
            # optimizer.step()
            # writer.add_scalar('D_loss', d_loss, n_iter)
            # print('Epoch : %d ;Iter : %d ; D_Loss : %0.4f '%(i, n_iter, d_loss))


            ############ CRITIC TRAINING ########################
            if n_iter % (n_critic+1) < n_critic: 
                requires_grad(model.a_encoder,flag=False)
                requires_grad(model.v_encoder,flag=False)
                requires_grad(model.generator,flag=False)
                requires_grad(model.discriminator,flag=True)

                gen_images,gen_pred,target_pred,gradients_img = model.forward_discriminator(source_video,target_audio,target_video)
                gen_images_up = F.upsample(gen_images, size=(224, 224), mode= 'bilinear')
                target_video_up = F.upsample(target_video, size=(224, 224), mode= 'bilinear')

                d_loss =   5 * w_loss(gen_pred,- torch.ones(gen_pred.size()).to(device)) +  5 * w_loss(target_pred,torch.ones(target_pred.size()).to(device)) + 1e-3 * gradients_img
                
                model.zero_grad()
                d_loss.backward()
                optimizer.step()
                writer.add_scalar('D_loss', d_loss, n_iter)
                print('Epoch : %d ;Iter : %d ; D_Loss : %0.4f '%(i, n_iter, d_loss))
            else:
                ########### GENERATOR TRAINING #####################

                requires_grad(model.a_encoder,flag=True)
                requires_grad(model.v_encoder,flag=True)
                requires_grad(model.generator,flag=True)
                requires_grad(model.discriminator,flag=False)

                gen_images,gen_pred = model.forward_generator(source_video,target_audio)
                gen_images_up = F.upsample(gen_images, size=(224, 224), mode= 'bilinear')
                target_video_up = F.upsample(target_video, size=(224, 224), mode= 'bilinear')

                cont_loss = loss_func(target_video_up,gen_images_up) 
                gen_loss =  10 * w_loss(gen_pred,torch.ones(gen_pred.size()).to(device))
                g_loss = cont_loss + gen_loss

                model.zero_grad()
                g_loss.backward()
                optimizer.step()

                writer.add_scalar('loss', g_loss, n_iter)

                print('Epoch : %d ; Iter : %d ; Content_Loss : %0.4f; Gen_Loss : %0.4f '%(i, n_iter, cont_loss,gen_loss))

        

            n_iter+= 1

            
        if not os.path.isdir('./video'):
            os.mkdir('./video')
        write_video(gen_images_up.cpu().detach().numpy().swapaxes(1,2).swapaxes(2,3),'./video/gen_%s.avi'%(n_iter),size=(224,224),name='gen')
        write_video(source_video.cpu().detach().numpy().swapaxes(1,2).swapaxes(2,3),'./video/source_%s.avi'%(n_iter),size=(size,size),name='source')            

        if i%10==9:
            save_model(optimizer,model,'./save_model',i//10)
        
        scheduler.step(g_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int,default=128,help="Size of image input")
    parser.add_argument("--style_dim", type=int,default=32,help="dimension of style input")
    parser.add_argument("--batchsize", type=int,default=1,help="Batch size")
    parser.add_argument("--epoch", type=int,default=100000,help="MNumber of epoch")
    parser.add_argument("--n_worker", type=int,default=0,help="MNumber of workers")
    parser.add_argument("--max_len", type=int,default=1,help="Length of sequence")
    parser.add_argument("--half_window", type=int,default=1200,help="length of half window")
    parser.add_argument("--load_pretrained", type=bool,default=False,help="load_pretrained model or not")
    parser.add_argument("--model_path", type=str,default='./save_model/model_1.pth',help="Link to save model")
    parser.add_argument("--warm_up", type=bool,default=False,help="Load part of pretrain model")


    args = parser.parse_args()

    train(size=args.size,style_dim=args.style_dim,batchsize=args.batchsize, epoch= args.epoch, n_worker=args.n_worker, max_len=args.max_len,half_window=args.half_window,load_pretrained=args.load_pretrained,model_path=args.model_path,warm_up=args.warm_up)