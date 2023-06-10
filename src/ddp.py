
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms 
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
from data import DatasetLoader
from tqdm import tqdm
#https://github.com/aju22/DDPM/blob/main/Denoising_Diffusion_Probabilistic_Model_(DDPMs).ipynb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

def linear_scheduler(timesteps, start=0.0001, end=0.02):
    
    """
    Returns linear schedule for beta
    """
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    
    """ 
    Returns values from vals for corresponding timesteps
    while considering the batch dimension.
    
    """
    batch_size = t.shape[0]
    output = vals.gather(-1, t.cpu())
    return output.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it after adding noise t times.
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        
        
        h = self.bn1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        
        h = self.bn2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeds = math.log(10000) / (half_dim - 1)
        embeds = torch.exp(torch.arange(half_dim, device=device) * -embeds)
        embeds = time[:, None] * embeds[None, :]
        embeds = torch.cat((embeds.sin(), embeds.cos()), dim=-1)
        return embeds


class Unet(nn.Module):
    """
    A simplified Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                PositionalEncoding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([ConvBlock(down_channels[i], down_channels[i+1], 
                                    time_emb_dim) for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([ConvBlock(up_channels[i], up_channels[i+1],
                                        time_emb_dim, up=True) for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        
        # Embedd time
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)
    



def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    # noise_pred = model(torch.unsqueeze(x_noisy,0), t)
    # return F.l1_loss(torch.unsqueeze(noise), noise_pred)

    noise_pred = model(x_noisy, t)

    return F.mse_loss(noise, noise_pred)
    #return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * MODEL(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            show_tensor_image(img.detach().cpu())


def load_transformed_dataset():
    '''
    Returns data after applying appropriate transformations,
    to work with diffusion models.
    '''
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform)

    test = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image):

    '''
    Plots image after applying reverse transformations.
    '''

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    plt.show()

    tra = transforms.ToTensor()
    return (tra(reverse_transforms(image)))






from torch.optim import Adam

# Define beta schedule
T = 300
betas = linear_scheduler(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


EPOCHS = 100
IMG_SIZE = 256
BATCH_SIZE = 8


def ddptrain(epochs):
    for epoch in range(epochs):
        print("epoch = "+str(epoch))
        los=[]
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch, t)
            loss.backward()
            optimizer.step()
            
            los.append(loss.item())

        if epoch%10==0:
            torch.save(model.state_dict(), "unet_%d.pt" %(epoch) )


        print(f"Epoch {epoch} | step {step:03d} Loss: {sum(los)/len(los)} ")
        #sample_plot_image()
    torch.save(model.state_dict(), "unet_final.pt" )
def forward(image,batch):
    t = torch.full((batch,), 299, device=device, dtype=torch.long)
    image, noise = forward_diffusion_sample(image, t)
    return image

def denoise(image):
    with torch.no_grad():
        for i in range(0,T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            image=sample_timestep(torch.tensor(image,device=device),t)
            #show_tensor_image(image.detach().to('cpu'))

        #t1=show_tensor_image(image.detach().to('cpu'))
    return image

MODEL = Unet()
MODEL.load_state_dict(torch.load("unet_final.pt"))
MODEL.to(device)


if __name__ == "__main__":

    #dataloader=DatasetLoader("./dataset/train/good",batch=BATCH_SIZE,size=IMG_SIZE)
    model = Unet()
    model.load_state_dict(torch.load("unet_final.pt"))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=3e-4)

    #test
    dataloader=DatasetLoader("./dataset/train/crack",batch=BATCH_SIZE,size=IMG_SIZE)
    image = next(iter(dataloader))[0]
    t0=show_tensor_image(image)
    image=torch.unsqueeze(image,0)

    ## add noise
    num_images = 10
    stepsize = int(T/num_images)
    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)

        image, noise = forward_diffusion_sample(image, t)
        #show_tensor_image(image)
    #show_tensor_image(image)

    model.eval()
    with torch.no_grad():
        for i in range(0,T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            image=sample_timestep(torch.tensor(image,device=device),t)
            #image=sample_timestep(torch.tensor(img,device=device),t)
        t1=show_tensor_image(image.detach().to('cpu'))
    



    tra = transforms.ToTensor()
    re=tra(t1)-tra(t0)
    show_tensor_image(torch.tensor(re))


