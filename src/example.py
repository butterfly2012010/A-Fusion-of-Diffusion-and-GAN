import ddp
from ddp import Unet
from data import DatasetLoader
import torch
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





#for train
# ddp.DATALOAD=DatasetLoader("./dataset/train/good",batch=ddp.BATCH_SIZE,size=ddp.IMG_SIZE)
# epochs = 100
# ddp.T=300
# model = Unet()
# ddp.OPTIMIZER = Adam(model.parameters(), lr=3e-4)
# model.load_state_dict(torch.load("unet_final.pt"))
# model.to(device)
# ddp.ddptrain(epochs)


########################
# # LOAD MODEL
# ddp.MODEL.load_state_dict(torch.load("unet_final.pt"))
# ddp.MODEL.to(device)

# # 測試
#dataloader=DatasetLoader("./dataset/train/good",batch=ddp.BATCH_SIZE,size=ddp.IMG_SIZE)
# image = next(iter(dataloader))[0]
# image=torch.unsqueeze(image,0)
# t0=ddp.show_tensor_image(image.detach().to('cpu'))

# # 加躁的圖 BATCH 要設定當前的圖片數量(len(image))
# image=ddp.forward(image,batch=1)


# # 有noisy的圖    
# #i_t=ddp.show_tensor_image(image)

# # 去躁
# image=ddp.denoise(image)
# i_0=ddp.show_tensor_image(image.detach().to('cpu'))

# print(i_0-t0)


