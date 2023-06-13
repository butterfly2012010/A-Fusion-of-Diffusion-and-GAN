import torch
import torch.nn as nn
from functools import reduce
from operator import __add__
from torchvision import transforms 
class ResBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

        self.kernel_sizes = (3, 3)
        self.conv_padding = reduce(__add__, 
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_sizes[::-1]])

    def forward(self, input):
        residual = input
        x = nn.ZeroPad2d(self.conv_padding)(input)
        x = self.bn1(self.conv1(x))
        x = self.prelu(x)
        x = nn.ZeroPad2d(self.conv_padding)(input)
        x = self.bn2(self.conv2(x))
        x += residual
        return x

class NoiseGenerator(nn.Module):
    def __init__(self, block, num_of_resblock=5, input_channels=3):
        super(NoiseGenerator, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 3, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.layers = self._make_layer(block, num_of_resblock)

        self.conv2 = nn.Conv2d(input_channels, 3, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(input_channels, 3, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(3)
        self.relu3 = nn.ReLU(inplace=True)

        self.kernel_sizes = (3, 3)
        self.conv_padding = reduce(__add__, 
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_sizes[::-1]])

    def _make_layer(self, block, num_of_resblock):
        layers = []
        for i in range(num_of_resblock):
          layers.append(block(in_channels=3, out_channels=3))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = nn.ZeroPad2d(self.conv_padding)(input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        residual = x

        x = self.layers(x)

        x = nn.ZeroPad2d(self.conv_padding)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        comb = x + residual

        comb = nn.ZeroPad2d(self.conv_padding)(comb)
        comb = self.conv3(comb)
        comb = self.bn3(comb)
        comb = self.relu3(comb)

        return comb
    

import torch
import torch.nn as nn
from functools import reduce
from operator import __add__

class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, stride=2):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        x = self.bn(self.conv(input))
        # print(x.shape)
        x = self.lrelu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, block, input_channels=3):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 3, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.channels = [64, 128, 128, 256, 256, 512, 512]
        self.strides = [2, 1, 2, 1, 2, 1, 2]
        self.layers = self._make_layer(block, self.channels, self.strides)

        self.flat = nn.Flatten()
        self.dense = nn.LazyLinear (out_features=1024)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.dense2 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()


    def _make_layer(self, block, chennels_num, strides):
        layers = []
        layers.append(block(in_channels=3, out_channels=chennels_num[0], stride = strides[0]))
        for i in range(1,len(chennels_num)-1):
          layers.append(block(in_channels=chennels_num[i-1], out_channels=chennels_num[i], stride = strides[i]))
        layers.append(block(in_channels=chennels_num[-2], out_channels=chennels_num[-1], stride = strides[-1]))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        print(x.shape)

        x = self.layers(x)
        print(x.shape)
        x = self.flat(x)
        x = self.dense(x)
        x = self.relu2(x)
        x = self.dense2(x)
        x = self.sig(x)


        return x
    
import ddp
from ddp import Unet
from data import DatasetLoader
import torch
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
ddp.MODEL.load_state_dict(torch.load("unet_final.pt"))
ddp.MODEL.to(device)
ddp.BATCH_SIZE=16
NormalDataloader = DatasetLoader("./data/dataset_same_size/train/goods",batch=ddp.BATCH_SIZE,size=ddp.IMG_SIZE)
AnomalyDataloader = DatasetLoader("./data/dataset_same_size/train/dot",batch=ddp.BATCH_SIZE,size=ddp.IMG_SIZE)




# Instantiate generator and discriminator
generator = NoiseGenerator(ResBlock)
discriminator = Discriminator(ConvLayer)
forward = ddp.forward
reverse = ddp.denoise

# Specify loss functions
gan_loss = nn.BCELoss()
aux_loss = nn.MSELoss()  # Change to the appropriate loss function for your task

# Set up optimizers
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0005)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0005)

#generator.load_state_dict(torch.load("generator_0.pt"))
#discriminator.load_state_dict(torch.load("discrimina_0.pt"))
# Training loop
num_epochs = 100
for epoch in range(0,num_epochs):
    los1=[]
    los2=[]
    los3=[]
    for Anomaly, Normal in zip(AnomalyDataloader, NormalDataloader): #normal 不夠的話請用data augmentation補充到跟所有anomaly相同的數量(或是刪掉幾張anomaly)
        #print(Anomaly.shape)
        # ============= setting ==========
        batch_size = Anomaly.size(0)

        # Generate fake images
        ### resize Anomaly to 64
        down_trans = transforms.Compose([
            transforms.Resize((64, 64))
        ])
        Anomaly = torch.stack([down_trans(img) for img in Anomaly])
        generated_noise = generator(Anomaly)
        Noised_anomaly = generated_noise + Anomaly

        # Generate noised normal images
        #Noised_normal = forward(Normal, batch=ddp.BATCH_SIZE)
        Noised_normal = forward(Normal, batch=batch_size)

        # Generate Denoised images
        Denoised_normal = reverse(Noised_normal)
        ### resize Noised_anomaly to 256
        up_trans = transforms.Compose([
            transforms.Resize((256, 256))
        ])
        Noised_anomaly = torch.stack([up_trans(img) for img in Noised_anomaly])
        Denoised_anomaly = reverse(Noised_anomaly)


        # =========== Training ==========
        # Train the discriminator
        discriminator_optimizer.zero_grad()
        
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        real_outputs = discriminator(Noised_anomaly)
        fake_outputs = discriminator(Noised_normal)
        
        discriminator_loss = gan_loss(real_outputs, real_labels) + gan_loss(fake_outputs, fake_labels)
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()
        
        # Train the generator
        generator_optimizer.zero_grad()
        
        fake_outputs = discriminator(Noised_anomaly)
        
        generator_loss = gan_loss(fake_outputs, real_labels)
        ## generator_loss.backward()
        ## generator_optimizer.step()
        
        # Compute auxiliary loss
        auxiliary_loss = aux_loss(Denoised_normal, Denoised_anomaly)  # Modify according to your task
        
        # Update generator parameters again with auxiliary loss
        ## generator_optimizer.zero_grad()
        (generator_loss + auxiliary_loss).backward(retain_graph=True)
        generator_optimizer.step()
        
        # Print losses or other metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {generator_loss.item()}, "
              f"Discriminator Loss: {discriminator_loss.item()}, Auxiliary Loss: {auxiliary_loss.item()}")
        los1.append(generator_loss.item())
        los2.append(discriminator_loss.item())
        los3.append(auxiliary_loss.item())

    logger = open('training_2.txt', 'a')
    logger.write('%d %f %f %f \n'%(epoch,sum(los1)/len(los1),sum(los2)/len(los2),sum(los3)/len(los3)))
    logger.close()
    if epoch%5==0:
        torch.save(generator.state_dict(), "generator2_%d.pt" %(epoch) )    
        torch.save(discriminator.state_dict(), "discrimina2_%d.pt" %(epoch) )
    if epoch%10==0:
        torch.save(ddp.MODEL.state_dict(), "unet2_%d.pt"%(epoch) )
torch.save(generator.state_dict(), "generator2_final.pt" )    
torch.save(discriminator.state_dict(), "discrimina2_final.pt"  )
torch.save(ddp.MODEL.state_dict(), "unet2_final.pt" )
