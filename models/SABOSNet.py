# %%
import random
import os
import os.path
import numpy as np
import math
import tqdm
import random
import copy
import torchio

import torch as t
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F 

import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


# %%

device = "cuda:2"
device1 = "cuda:2"
device2 = "cuda:2"

train_path = "../../train_data_tcia.pth"
test_path = "../../test_data_tcia.pth"

epoch150_path_name = "swap10_tcia_120.pth"
epoch150_50_path_name = "swap10_tcia_120+80.pth"

SSL_path_name = "genesisSC500.pth"
SSL_path_name1 = "swap10.pth"
SSL_path_name2 = "swap5.pth"
file_name = "swap10_dice_log_tcia.txt"
file_name_1 = "swap10_loss_log_tcia.txt"
file_name_2 = "swap10_totalDice_log_tcia.txt"

y_pred_path = "y_pred_swap10_tcia.npy"
y_true_path = "y_true_swap10_tcia.npy"

epoch1 = 80
epoch2 = 40
ssl_epoch_1, ssl_epoch_2 = 0, 500
ssl_epoch_3, ssl_epoch_4 = 0, 50
ssl_epoch_5, ssl_epoch_6 = 50, 100

# # Encoder - Decoder Network

# %%


class ChannelPool(nn.Module):
    def forward(self,x):
        return t.cat((t.max(x,1)[0].unsqueeze(1), t.mean(x,1).unsqueeze(1)),dim=1)

class SpatialGate(nn.Module):

    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv3d(2,1,kernel_size=7,stride=1,padding=3, bias=False),
            nn.BatchNorm3d(1,eps=1e-5,momentum=0.01,affine=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        xcompress=self.compress(x)
        spatialAttention=self.conv(xcompress)
        return x*spatialAttention
    
class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.size(0),-1)

class ChannelGate(nn.Module):
  
    def __init__(self,channels,reductionRatio=16,poolTypes=['avg','max']):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channels,channels//reductionRatio),
            nn.ReLU(),
            nn.Linear(channels//reductionRatio,channels)
        )
        self.poolTypes = poolTypes
  
    def forward(self,x):
        attentionSum = None
        for poolType in self.poolTypes:
            if poolType=='avg':
                avgPool = F.avg_pool3d(x,(x.size(2),x.size(3),x.size(4)),stride=(x.size(2),x.size(3),x.size(4)))
                channelAttention = self.mlp(avgPool)
            if poolType=='max':
                maxPool = F.max_pool3d(x,(x.size(2),x.size(3),x.size(4)),stride=(x.size(2),x.size(3),x.size(4)))
                channelAttention = self.mlp(maxPool)
        if attentionSum is None:
              attentionSum = channelAttention
        else:
              attentionSum=attentionSum+channelAttention
    
        scale=t.sigmoid(attentionSum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x*scale


class CBAM(nn.Module):
  
    def __init__(self,channels,reductionRatio=16,poolTypes=['avg','max']):
        super().__init__()
        self.channelGate = ChannelGate(channels,reductionRatio,poolTypes)
        self.spatialGate = SpatialGate()

    def forward(self,x):
        x=self.spatialGate(x)
        x=self.channelGate(x)
        return x 
    
class CBAMResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels,kernelSize=3):
        super().__init__()
        self.resnetblock1 = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=kernelSize,padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
            )
        self.resnetblock2 = nn.Sequential(
            nn.Conv3d(out_channels,out_channels,kernel_size=kernelSize,padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels,out_channels,kernel_size=kernelSize,padding=1),
            nn.BatchNorm3d(out_channels)
            )
        self.CBAM = CBAM(out_channels,16,['avg','max'])
        self.reluUnit = nn.Sequential(
            nn.ReLU(inplace=False)
            )

    def forward(self,x):
        x=self.resnetblock1(x)
        x1=x
        x=self.resnetblock2(x)
        x=self.CBAM(x)
        x=x+x1
        x=self.reluUnit(x)
        return x
    
class GSEncoder(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.downConv  = nn.Sequential(
            nn.Conv3d(1,32,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=False),
        )

        self.layer21 = nn.Sequential(
            CBAMResnetBlock(32,40),
            CBAMResnetBlock(40,40)
        )

        self.layer22 = nn.Sequential(
            CBAMResnetBlock(40,48),
            CBAMResnetBlock(48,48)
        )

        self.layer23 = CBAMResnetBlock(48,56)

    def forward(self,x):

        x0=self.downConv(x)
        x1=self.layer21(x0)
        x2=self.layer22(x1)
        x3=self.layer23(x2)
        #print(x3.shape)
        return x,x0,x1,x2,x3


class GSDecoder_part1(nn.Module):
    
    def __init__(self):
        super().__init__()
    
        self.layer23 = nn.Sequential(
            CBAMResnetBlock(56,56),
            CBAMResnetBlock(56,48)
        )
        self.layer22 = nn.Sequential(
            CBAMResnetBlock(96,48),
            CBAMResnetBlock(48,48),
            CBAMResnetBlock(48,40)
        )
        self.layer21 = nn.Sequential(
            CBAMResnetBlock(80,40),
            CBAMResnetBlock(40,40),
            CBAMResnetBlock(40,32)
        )
        self.layer20 = nn.Sequential(
            CBAMResnetBlock(64,32),
            CBAMResnetBlock(32,32)
        )
        self.transposeConv = nn.ConvTranspose3d(32,16,kernel_size=2,stride=2)
        self.layer10 = nn.Conv3d(17,16,kernel_size=3,padding=1)

    def forward(self,x,x0,x1,x2,x3):
        
        y=self.layer23(x3)
        y=t.cat([y,x2],dim=1)
        y=self.layer22(y)
        y=t.cat([y,x1],dim=1)
        y=self.layer21(y)
        y=t.cat([y,x0],dim=1)
        y=self.layer20(y)
        y=self.transposeConv(y)
        y=t.cat([y,x],dim=1)
        y=self.layer10(y)
        #print(y.shape)
        return y


class GSNetCore(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = GSEncoder().to(device1)
        self.decoder_part1 = GSDecoder_part1().to(device2)
  
    def forward(self,x):
        x,x0,x1,x2,x3=self.encoder(x)
        y=self.decoder_part1(x.to(device2),x0.to(device2),x1.to(device2),x2.to(device2),x3.to(device2))
        return y


class GSNetModel(nn.Module):
  
    def __init__(self):
        super().__init__()
        self.core = GSNetCore()
        self.finalConv = nn.Sequential(
            nn.Conv3d(16,8,1),
            nn.Softmax(dim=1),
        ).to(device2)
  
    def forward(self,x):
        y=self.core(x)
        y=self.finalConv(y)
        #print(y.shape)
        return y
    
class PreTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = GSNetCore()
        self.finalConv = nn.Sequential(
                nn.Conv3d(16,1,1),
                nn.Sigmoid()
                #nn.Conv3d(16,10,1),
                #nn.Softmax(dim=1),
                )
        
    def forward(self,x):
        y=self.core(x)
        y=self.finalConv(y)
        #print(y.shape)
        return y

# # Proxy tasks

# ### Flip
    
def flip(x,prob=0.5):
    
    count = 3
    while count > 0 and random.random() < prob:
        deg = random.choice([0, 1, 2])
        x = np.flip(x, axis=deg)
        count -= 1
    
    return x


# ### Non-linear transformation

# %%


def nonlinear_transformation(img, prob=0.5):
    
    if random.random() >= prob:
        return img
    
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1,1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]

    t = np.linspace(0.0, 1.0, 100000)
    bezier_curve = np.array([ 1.0*(t**3), 3.0*(t**2)*(1-t), 3.0*t*((1-t)**2), 1.0*((1-t)**3)])
    xvals = np.dot(np.array(xpoints), bezier_curve)
    yvals = np.dot(np.array(ypoints), bezier_curve)
    

    if random.random() < 0.5:
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    
    transformed_img = np.interp(img, xvals, yvals)
    return transformed_img


# ### Local Pixel shuffling

# %%


def pixel_shuffling(img, prob=0.5):
    
    if random.random() >= prob:
        return img
    
    shuffled_img = copy.deepcopy(img)
    real_img = copy.deepcopy(img)
    channel, z, x, y = img.shape
    
    for _ in range(10000):
        size_z = random.randint(1, z//10)
        size_x = random.randint(1, x//10)
        size_y = random.randint(1, y//10)
        
        dim_z = random.randint(0, z-size_z)
        dim_x = random.randint(0, x-size_x)
        dim_y = random.randint(0, y-size_y)
        
        window = real_img[:, dim_z:dim_z+size_z, dim_x:dim_x+size_x, dim_y:dim_y+size_y]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((size_z, size_x, size_y))
        
        shuffled_img[:, dim_z:dim_z+size_z, dim_x:dim_x+size_x, dim_y:dim_y+size_y] = window
        
    return shuffled_img


# ### Outpainting

# %%


def outpainting(img):
    
    outpainted_img = copy.deepcopy(img)
    channel, z, x, y = img.shape
    outpainted_img = np.random.rand(channel, z, x, y) * 1.0
    
    count = 5
    while count > 0 and (random.random() < 0.95 or count == 5):
        size_z = z - random.randint(3*z//7, 4*z//7)
        size_x = x - random.randint(3*x//7, 4*x//7)
        size_y = y - random.randint(3*y//7, 4*y//7)
        
        dim_z = random.randint(3, z-size_z-3)
        dim_x = random.randint(3, x-size_x-3)
        dim_y = random.randint(3, y-size_y-3)
        
        outpainted_img[:, dim_z:dim_z+size_z, dim_x:dim_x+size_x, dim_y:dim_y+size_y] = img[:, dim_z:dim_z+size_z, dim_x:dim_x+size_x, dim_y:dim_y+size_y]
        count -= 1
    
    return outpainted_img


# ### Inpainting

# %%


def inpainting(img):
    
    inpainted_img = copy.deepcopy(img)
    channel, z, x, y = img.shape
    
    count = 5
    while count > 0 and random.random() < 0.95:
        size_z = random.randint(z//6, z//3)
        size_x = random.randint(x//6, x//3)
        size_y = random.randint(y//6, y//3)
        
        dim_z = random.randint(3, z-size_z-3)
        dim_x = random.randint(3, x-size_x-3)
        dim_y = random.randint(3, y-size_y-3)
        
        inpainted_img[:, dim_z:dim_z+size_z, dim_x:dim_x+size_x, dim_y:dim_y+size_y] = np.random.rand(size_z, size_x, size_y) * 1.0
        count -= 1
        
    return inpainted_img


# # Dataset Preparation

# ### Train dataloader

# %%


# ### Self Supervision dataloader

# %%


class RandomCrop3D(object):

    def __init__(self, crop_size, hu_min=-1000.0, hu_max=1000.0, airgap_threshold=0.10):
        
        if isinstance(crop_size, tuple): 
            self.crop_size = crop_size
        else:
            self.crop_size = (crop_size, crop_size, crop_size)
            
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.airgap_threshold = airgap_threshold
    
    def crop_3d_volume(self,img_tensor, crop_dim, crop_size):

        full_dim1, full_dim2, full_dim3 = img_tensor.shape
        z_crop, x_crop, y_crop = crop_dim
        dim1, dim2, dim3 = crop_size

        # check if crop size matches image dimensions
        if full_dim1 == dim1:
            img_tensor = img_tensor[:, x_crop:x_crop + dim2, y_crop:y_crop + dim3]
        elif full_dim2 == dim2:
            img_tensor = img_tensor[z_crop:z_crop + dim1, :, y_crop:y_crop + dim3]
        elif full_dim3 == dim3:
            img_tensor = img_tensor[z_crop:z_crop + dim1, x_crop:x_crop + dim2, :]
        # standard crop
        else:
            img_tensor = img_tensor[z_crop:z_crop + dim1, x_crop:x_crop + dim2, y_crop:y_crop + dim3]
        
        return img_tensor

    def find_random_crop_dim(self,full_vol_dim, crop_size):
        
        assert full_vol_dim[0] >= crop_size[0], f"{full_vol_dim, crop_size} Exceeded crop size along z axis"
        assert full_vol_dim[1] >= crop_size[1], f"{full_vol_dim, crop_size} Exceeded crop size along x axis"
        assert full_vol_dim[2] >= crop_size[2], f"{full_vol_dim, crop_size} Exceeded crop size along y axis"

        if full_vol_dim[0] == crop_size[0]:
            z_crop = crop_size[0]
        else:
            z_crop = np.random.randint(full_vol_dim[0] - crop_size[0])

        if full_vol_dim[1] == crop_size[1]:
            x_crop = crop_size[1]
        else:
            x_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

        if full_vol_dim[2] == crop_size[2]:
            y_crop = crop_size[2]
        else:
            y_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

        return (z_crop, x_crop, y_crop)
    
    def __call__(self, img):
        
        img_tensor = copy.deepcopy(img)
        img_tensor = img_tensor[30:, 50:, :]
        
        #Clip Hounsfield units of CT and normalize between [0,1] (already done during preprocessing)
        #img_tensor[img_tensor < self.hu_min] = self.hu_min
        #img_tensor[img_tensor > self.hu_max] = self.hu_max
        #img_tensor = 1.0*(img_tensor-self.hu_min) / (self.hu_max-self.hu_min)
        
        #Crop CTs until air gaps are less than a threshold
        while True:
            crop_dim = self.find_random_crop_dim(img_tensor.shape, self.crop_size) 
            cropped_img = self.crop_3d_volume(img_tensor, crop_dim, self.crop_size)
            cropped_img_dims = cropped_img.shape
            
            air_gap = 0
            for voxal in cropped_img.flatten():
                if voxal==0.0:
                    air_gap += 1
            if air_gap/(cropped_img_dims[0]*cropped_img_dims[1]*cropped_img_dims[2]) > self.airgap_threshold: 
                continue
            
            return cropped_img.reshape(1,cropped_img_dims[0],cropped_img_dims[1],cropped_img_dims[2])


# %%


class Augumentation:

    def __init__(self, flip_rate=0.4 , nonlinear_rate=0.9 , shuffle_rate=0.5 , painting_rate=0.9 , outpainting_rate=0.8): #inpaint_rate = 1- outpaint_rate
        
        self.flip_rate = flip_rate
        self.nonlinear_rate = nonlinear_rate
        self.shuffle_rate = shuffle_rate
        self.painting_rate = painting_rate
        self.outpainting_rate = outpainting_rate
        self.inpainting_rate = 1.0 - self.outpainting_rate
        
    def __call__(self, real_img):
        
        img_copy = copy.deepcopy(real_img)
        
        # Flip
        augumented_img = flip(img_copy, self.flip_rate)

        # Local Shuffle Pixel
        augumented_img = pixel_shuffling(augumented_img, self.shuffle_rate)

        # Non-Linear transformation
        augumented_img = nonlinear_transformation(augumented_img, self.nonlinear_rate)

        # Inpainting & Outpainting
        if random.random() < self.painting_rate:
            if random.random() < self.inpainting_rate:
                # Inpainting
                augumented_img = inpainting(augumented_img)
            else:
                # Outpainting
                augumented_img = outpainting(augumented_img)
            
        return real_img.copy(), augumented_img.copy()


# %%


class SelfSupervisionDataset(Dataset):
    
    def __init__(self, root_dir=None, transform=None):
        self.path = root_dir
        self.transform = transform
        self.datas = t.load(self.path)
   
    def __getitem__(self,index):
        dataset = self.datas[index]
        data = dataset['img']
        arr = data.numpy().astype(np.float32)
        
        if self.transform:
            tup = self.transform(arr)
            
        return tup
    
    def __len__(self):
        return len(self.datas)

    
transform = transforms.Compose([
                    RandomCrop3D((32, 128, 128)), # z x y
                    Augumentation(flip_rate=0.4 , nonlinear_rate=0.9 , shuffle_rate=0.5 , painting_rate=0.9 , outpainting_rate=0.8) #inpainting_rate = 1 - outpainting_rate
                    ])

dataset = SelfSupervisionDataset(train_path, transform=transform)

batch_size = 12
pretrain_dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
print(len(pretrain_dataloader))

pretrain_model = PreTrainModel()
pretrain_model = pretrain_model.to(device1)

Loss = nn.MSELoss()
optimizer = t.optim.SGD(pretrain_model.parameters(), 0.1, momentum=0.9, weight_decay=0.0, nesterov=False)


for epoch in range(ssl_epoch_1, ssl_epoch_2):
    i=0
    j=0
    train_loss = 0
    
    print("epoch", epoch+1)
    for x_train, x_transformed in tqdm.tqdm(pretrain_dataloader):
        i+=1
        
        #print(f'epoch {epoch+1} batch {i} tensor_shape {x_train.shape}')
        
        out = pretrain_model(x_transformed.type(t.FloatTensor).to(device))
        
        loss = Loss(out, x_train.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #print("Batch loss: ",loss.item())
        
        if (epoch+1)%250==0:
            with t.no_grad():
                np.save("real_"+str(epoch+1)+".npy", x_train.cpu())
                np.save("deformed_"+str(epoch+1)+".npy",x_transformed.cpu())
                np.save("reconstructed_"+str(epoch+1)+".npy", out.cpu())

    if (epoch+1)%10==0:
        t.save({
                'epoch': epoch,
                'model_state_dict': pretrain_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, SSL_path_name)
        
        with open(file_name, "a") as f:
            f.write(f"epoch {epoch+1}")
            f.write(f"\tLoss: {train_loss/(len(pretrain_dataloader)-j)}\n")
        
    #scheduler.step(epoch)
    
    print("\nLoss: ",train_loss/(len(pretrain_dataloader)-j),'\n')



"""loaded_model = t.load("genesisSC.pth", map_location="cpu")
print(loaded_model["epoch"])
pretrain_model.load_state_dict(loaded_model["model_state_dict"])
#optimizer.load_state_dict(loaded_model["optimizer_state_dict"])"""

class SelfSupervisionDataset(Dataset):
    
    def __init__(self, root_dir=None, transform=None):
        self.path = root_dir
        self.transform = transform
        self.datas = t.load(self.path)
   
    def __getitem__(self,index):
        dataset = self.datas[index]
        data = dataset['img']
        arr = data.numpy().astype(np.float32)
        #print(arr.shape)
        
        if self.transform is not None:
            arr = self.transform(arr.reshape(1,arr.shape[0],arr.shape[1],arr.shape[2]))
            #print(arr.shape)
        
        arr = arr[:, 30:, 50:, :]
        #print(arr.shape)
        return arr
    
    def __len__(self):
        return len(self.datas)

transformations = torchio.transforms.Compose([
                            torchio.transforms.RandomNoise(std=0.0001),
                            torchio.transforms.RandomElasticDeformation(),
                          ])

dataset = SelfSupervisionDataset(train_path, transform=transformations)

batch_size=1
pretrain_dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)

optimizer = t.optim.Adam(pretrain_model.parameters())
Loss = t.nn.MSELoss()
aug = torchio.transforms.RandomSwap(10,2000)

for epoch in range(ssl_epoch_3, ssl_epoch_4):
    i=0
    j=0
    train_loss = 0
    print("epoch ", epoch+1)
    for x_train in tqdm.tqdm(pretrain_dataloader):
        i+=1
        #print(x_train.shape)
        #x_train = x_train.reshape(1,1,x_train.shape[1], x_train.shape[2], x_train.shape[3])
        #print(f'epoch {epoch+1} batch {i} tensor_shape {x_train.shape}')
        
        x_transformed = aug(x_train.reshape(1,x_train.shape[2],x_train.shape[3],x_train.shape[4]))
        x_transformed = x_transformed.reshape(1,1,x_transformed.shape[1],x_transformed.shape[2],x_transformed.shape[3]).to(device)
        out = pretrain_model(x_transformed)
        
        loss = Loss(out, x_train.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #print("Batch loss: ",loss.item())
        
        if (epoch+1)%50==0:
            with t.no_grad():
                np.save("real_"+str(epoch+1)+".npy", x_train.cpu())
                np.save("deformed_"+str(epoch+1)+".npy",x_transformed.cpu())
                np.save("reconstructed_"+str(epoch+1)+".npy", out.cpu())

    if (epoch+1)%10==0:
        t.save({
                'epoch': epoch,
                'model_state_dict': pretrain_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, SSL_path_name1)
        
        with open(file_name, "a") as f:
            f.write(f"epoch {epoch+1}")
            f.write(f"\tLoss: {train_loss/(len(pretrain_dataloader)-j)}\n")
        
    #scheduler.step(epoch)
    
    print("\nLoss: ",train_loss/(len(pretrain_dataloader)-j),'\n')
    
optimizer = t.optim.Adam(pretrain_model.parameters())
Loss = t.nn.MSELoss()
aug = torchio.transforms.RandomSwap(5,25000)

for epoch in range(ssl_epoch_5, ssl_epoch_6):
    i=0
    j=0
    train_loss = 0
    print("epoch ", epoch+1)
    for x_train in tqdm.tqdm(pretrain_dataloader):
        i+=1
        #print(x_train.shape)
        #x_train = x_train.reshape(1,1,x_train.shape[1], x_train.shape[2], x_train.shape[3])
        #print(f'epoch {epoch+1} batch {i} tensor_shape {x_train.shape}')
        
        x_transformed = aug(x_train.reshape(1,x_train.shape[2],x_train.shape[3],x_train.shape[4]))
        x_transformed = x_transformed.reshape(1,1,x_transformed.shape[1],x_transformed.shape[2],x_transformed.shape[3]).to(device)
        out = pretrain_model(x_transformed)
        
        loss = Loss(out, x_train.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #print("Batch loss: ",loss.item())
        
        if (epoch+1)%50==0:
            with t.no_grad():
                np.save("real_"+str(epoch+1)+".npy", x_train.cpu())
                np.save("deformed_"+str(epoch+1)+".npy",x_transformed.cpu())
                np.save("reconstructed_"+str(epoch+1)+".npy", out.cpu())

    if (epoch+1)%10==0:
        t.save({
                'epoch': epoch,
                'model_state_dict': pretrain_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, SSL_path_name2)
        
        with open(file_name, "a") as f:
            f.write(f"epoch {epoch+1}")
            f.write(f"\tLoss: {train_loss/(len(pretrain_dataloader)-j)}\n")
        
    #scheduler.step(epoch)
    
    print("\nLoss: ",train_loss/(len(pretrain_dataloader)-j),'\n')

# # Dataset Preparation

# ### Train dataloader

# %%

class HaN_Dataset(Dataset):
    
    def __init__(self, root_dir=None, transform=False, alpha=1000, sigma=30, alpha_affine=0.04):
        super().__init__()
        self.path = root_dir
        self.datas = t.load(self.path)
        
        self.transform = transform
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
    
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy().astype(np.float32)
        
        if not self.transform:
            masklst = []
            for mask in data['mask']:
                if mask is None:
                    mask = np.zeros((1,img.shape[0],img.shape[1],img.shape[2])).astype(np.uint8)
                masklst.append(mask.astype(np.uint8).reshape((1,img.shape[0],img.shape[1],img.shape[2]))) 
            mask0 = np.zeros_like(masklst[0]).astype(np.uint8)
            for mask in masklst:
                mask0 = np.logical_or(mask0, mask).astype(np.uint8)
            mask0 = 1 - mask0
            return t.from_numpy(img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))), t.from_numpy(np.concatenate([mask0]+masklst, axis=0)), True
        
        im_merge = np.concatenate([img[...,None]]+[mask.astype(np.float32)[...,None] for mask in data['mask']], axis=3)
        # Apply transformation on image
        im_merge_t, new_img = self.elastic_transform3Dv2(im_merge,self.alpha,self.sigma,min(im_merge.shape[1:-1])*self.alpha_affine)
        # Split image and mask ::2, ::2, ::2
        im_t = im_merge_t[...,0]
        im_mask_t = im_merge_t[..., 1:].astype(np.uint8).transpose(3, 0, 1, 2)
        mask0 = np.zeros_like(im_mask_t[0, :, :, :]).reshape((1,)+im_mask_t.shape[1:]).astype(np.uint8)
        im_mask_t_lst = []
        flagvect = np.ones((8,), np.float32)
        retflag = True
        for i in range(7):
            im_mask_t_lst.append(im_mask_t[i,:,:,:].reshape((1,)+im_mask_t.shape[1:]))
            if im_mask_t[i,:,:,:].max() != 1: 
                retflag = False
                flagvect[i+1] = 0
            mask0 = np.logical_or(mask0, im_mask_t[i,:,:,:]).astype(np.uint8)
        if not retflag: flagvect[0] = 0
        mask0 = 1 - mask0
        return t.from_numpy(im_t.reshape((1,)+im_t.shape[:3])), t.from_numpy(np.concatenate([mask0]+im_mask_t_lst, axis=0)), flagvect
        
    def __len__(self):
        return len(self.datas)
    
    def elastic_transform3Dv2(self, image, alpha, sigma, alpha_affine, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.
         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         From https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
        """
        # affine and deformation must be slice by slice and fixed for slices
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = image.shape # image is contatenated, the first channel [:,:,:,0] is the image, the second channel 
        # [:,:,:,1] is the mask. The two channel are under the same tranformation.
        shape_size = shape[:-1] # z y x
        # Random affine
        shape_size_aff = shape[1:-1] # y x
        center_square = np.float32(shape_size_aff) // 2
        square_size = min(shape_size_aff) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        new_img = np.zeros_like(image)
        for i in range(shape[0]):
            new_img[i,:,:,0] = cv2.warpAffine(image[i,:,:,0], M, shape_size_aff[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
            for j in range(1, 8):
                new_img[i,:,:,j] = cv2.warpAffine(image[i,:,:,j], M, shape_size_aff[::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT, borderValue=0)
        dx = gaussian_filter((random_state.rand(*shape[1:-1]) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape[1:-1]) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape_size_aff[1]), np.arange(shape_size_aff[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        new_img2 = np.zeros_like(image)
        for i in range(shape[0]):
            new_img2[i,:,:,0] = map_coordinates(new_img[i,:,:,0], indices, order=1, mode='constant').reshape(shape[1:-1])
            for j in range(1, 8):
                new_img2[i,:,:,j] = map_coordinates(new_img[i,:,:,j], indices, order=0, mode='constant').reshape(shape[1:-1])
        return np.array(new_img2), new_img
# %%


traindataset = HaN_Dataset(train_path, transform=True)
traindataloader = DataLoader(traindataset, batch_size=1, shuffle=True)
testdataset = HaN_Dataset(test_path, transform=False)
testdataloader = DataLoader(testdataset, batch_size=1)

print(len(traindataloader),len(testdataloader))


# ### Self Supervision dataloader

# %%

# # Fine tuning

# %%


def crossentropy(y_pred, y_true, flagvec, device):
    retv = - t.sum(t.sum(t.sum(t.sum(t.log(t.clamp(y_pred,1e-6,1))*y_true.type(t.cuda.FloatTensor),4),3),2),0) * flagvec.to(device)
    return t.sum(retv / (t.sum(t.sum(t.sum(t.sum(y_true.type(t.cuda.FloatTensor),4),3),2),0) + 1e-6)) / y_true.size()[1]


# %%


def tversky_loss_wmask(y_pred, y_true, flagvec, device):
    alpha = 0.5
    beta  = 0.5
    ones = t.ones_like(y_pred) 

    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true.type(t.cuda.FloatTensor)
    g1 = ones-g0
    num = t.sum(t.sum(t.sum(t.sum(p0*g0, 4),3),2),0) #(0,2,3,4)) #K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*t.sum(t.sum(t.sum(t.sum(p0*g1,4),3),2),0) + beta*t.sum(t.sum(t.sum(t.sum(p1*g0,4),3),2),0) #(0,2,3,4))

    T = t.sum((num* flagvec.to(device))/(den+1e-5))
    return t.sum(flagvec.to(device))-T


# %%


def focal(y_pred, y_true, flagvec, device):
    retv = - t.mean(t.mean(t.mean(t.mean(t.log(t.clamp(y_pred,1e-6,1))*y_true.type(t.cuda.FloatTensor)*t.pow(1-y_pred,2),4),3),2),0) * flagvec.to(device)
    return t.sum(retv)


# %%


def caldice(y_pred, y_true):

    y_pred = y_pred.data.cpu().numpy().transpose(1,0,2,3,4) # inference should be arg max
    y_pred = np.argmax(y_pred, axis=0).squeeze() # z y x
    y_true = y_true.data.cpu().numpy().transpose(1,0,2,3,4).squeeze() # .cpu()
    avgdice = []
    y_pred_1 = y_pred==1
    y_true_1 = y_true[1,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==2
    y_true_1 = y_true[2,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==3
    y_true_1 = y_true[3,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==4
    y_true_1 = y_true[4,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==5
    y_true_1 = y_true[5,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==6
    y_true_1 = y_true[6,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==7
    y_true_1 = y_true[7,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    """
    y_pred_1 = y_pred==8
    y_true_1 = y_true[8,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    
    y_pred_1 = y_pred==9
    y_true_1 = y_true[9,:,:,:]
    if y_pred_1.sum() + y_true_1.sum() == 0: avgdice.append(-1)
    else: avgdice.append(2.*(np.logical_and(y_pred_1, y_true_1).sum()) / (1.0*(y_pred_1.sum() + y_true_1.sum())))
    """
    for dice in avgdice: 
        if dice != -1:
            assert 0 <= dice <= 1
    return avgdice

def get_yPredDash(y_pred):
    
    y_pred = y_pred.data.cpu().numpy() # inference should be arg max
    y_pred_1 = np.zeros_like(y_pred).astype(np.uint8)
    y_pred = np.argmax(y_pred, axis=1).squeeze() # z y x
    
    for i in range(8):
        y_pred_1[:,i,:,:,:] = y_pred==i
    
    return y_pred_1

# %%

SABOSNet_Model = GSNetModel()

SABOSNet_Model.core.load_state_dict(pretrain_model.core.state_dict())

#lossweight = np.array([2.22, 1.06, 1.02, 1.74, 1.93, 1.93, 1.13, 1.15, 1.90, 1.98], np.float32)
lossweight = np.array([2.22, 1.06, 1.02, 1.74, 1.93, 1.93, 1.13, 1.15], np.float32)
ceweight = 0.05
focweight = 0.5

optimizer = t.optim.RMSprop(SABOSNet_Model.parameters(),lr = 2e-3)
#scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
"""
loaded_model = t.load(epoch150_path_name)
print(loaded_model["epoch"]+1)
SABOSNet_Model.load_state_dict(loaded_model["model_state_dict"])
optimizer.load_state_dict(loaded_model["optimizer_state_dict"])
"""
# %%


for epoch in range(epoch1):
    i=0
    j=0
    train_loss = 0
    print("epoch ", epoch+1)
    for x_train, y_train, flagvec in tqdm.tqdm(traindataloader):
        
        i+=1

        #print(f'epoch {epoch+1} batch {i} tensor_shape {x_train.shape}')
        
        """
        if x_train.shape[2] > 150:
            x_train = x_train[:,:,37:37+64,...]
            y_train = y_train[:,:,37:37+64,...]
        """ 
        if x_train.shape[-2] > 300:
            x_train = x_train[...,10:,:]
            y_train = y_train[...,10:,:]
        
        x_train = x_train.to(device1)
        out = SABOSNet_Model(x_train)
        
        out = out.to(device2)
        y_train = y_train.to(device2)
        loss = tversky_loss_wmask(out, y_train, flagvec*t.from_numpy(lossweight), device2)
        celoss = crossentropy(out, y_train, flagvec*t.from_numpy(lossweight), device2)
        floss = focal(out, y_train, flagvec*t.from_numpy(lossweight), device2)
        optimizer.zero_grad()
        (loss + (ceweight*celoss) + (focweight*floss)).backward()
        optimizer.step()
        train_loss += loss.item() + (ceweight*celoss).item() + (focweight*floss).item()

        y_train.detach_()
        out.detach_()
        del loss, x_train, y_train, out, floss, celoss
        
    print("train loss: ",train_loss/(len(traindataloader)-j),'\n')
    
    testtq = tqdm.tqdm(testdataloader)#, desc='loss', leave=True)
    test_loss = 0
    for x_test, y_test, flagvecTest in testtq:
        with t.no_grad():
            x_test = x_test.to(device1)
            y_test = y_test.to(device2)
            out = SABOSNet_Model(x_test)
            out = out.to(device2)
            loss = tversky_loss_wmask(out, y_test, flagvecTest*t.from_numpy(lossweight), device2)
            celoss = crossentropy(out, y_test, flagvecTest*t.from_numpy(lossweight), device2)
            floss = focal(out, y_test, flagvecTest*t.from_numpy(lossweight), device2)
            test_loss += loss.item() + (ceweight*celoss).item() + (focweight*floss).item()
            del out, x_test, y_test
            
    print(f"test loss: {test_loss/len(testdataloader)}")
    
    with open(file_name_1, "a") as f:
        f.write(f"epoch {epoch+1}")
        f.write(f"\ttrain loss: {train_loss/(len(traindataloader)-j)}")
        f.write(f"\ttest loss: {test_loss/len(testdataloader)}\n")
                        
    testloss = [0 for _ in range(7)]
    testtq = tqdm.tqdm(testdataloader, desc='loss', leave=True)
        
    for x_test, y_test, _ in testtq:
        with t.no_grad():
            
            x_test = x_test.to(device1)
            y_test = y_test.to(device2)
            o = SABOSNet_Model(x_test)
            o = o.to(device2)
            loss = caldice(o, y_test)
            testtq.set_description("test loss %f" % (sum(loss)/7))
            testtq.refresh() # to show immediately the update
            testloss = [l+tl if l != -1 else tl for l,tl in zip(loss, testloss)]  #testloss = [l+tl for l,tl in zip(loss, testloss)]
            del x_test, y_test, o
        
    testloss = [l / len(testtq) for l in testloss]
    with open(file_name_2, "a") as f:
        f.write(f"epoch {epoch+1}")
        f.write('\tDice coeff %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n' % tuple(testloss))

    if (epoch+1)%10==0:
        
        t.save({
        'epoch': epoch,
        'model_state_dict':  SABOSNet_Model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, epoch150_path_name)
        
        with open(file_name, "a") as f:
            f.write(f"epoch {epoch+1}")
            f.write('\tDice coeff %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n' % tuple(testloss))
        
        for x,y,_ in testdataloader:
            with t.no_grad():
                y_pred = SABOSNet_Model(x.to(device1))
                y_predDash = get_yPredDash(y_pred)
                np.save(y_pred_path,y_predDash)
                np.save(y_true_path,y.cpu())
                #print(y_pred.shape, y.shape) 
                break
            

    #scheduler.step(test_loss/len(testdataloader))

# %%


#with open(file_name, "a") as f:
#    f.write("\n\nfinal 150 epoch training\n\n")

#prev_lr = optimizer.param_groups[0]['lr']
optimizer = t.optim.SGD(SABOSNet_Model.parameters(),lr = 1e-3, momentum=0.9)
#scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

for epoch in range(epoch1,epoch1 + epoch2):
    i=0
    j=0
    train_loss = 0
    print("epoch ", epoch+1)
    for x_train, y_train, flagvec in tqdm.tqdm(traindataloader):
        #SABOSNet_Model.train()
        i+=1

        #print(f'epoch {epoch+1} batch {i} tensor_shape {x_train.shape}')
        
        """
        if x_train.shape[2] > 150:
            x_train = x_train[:,:,37:37+64,...]
            y_train = y_train[:,:,37:37+64,...]
        """ 
        if x_train.shape[-2] > 300:
            x_train = x_train[...,10:,:]
            y_train = y_train[...,10:,:]
        
        x_train = x_train.to(device1)
        out = SABOSNet_Model(x_train)
        
        out = out.to(device2)
        y_train = y_train.to(device2)
        loss = tversky_loss_wmask(out, y_train, flagvec*t.from_numpy(lossweight), device2)
        celoss = crossentropy(out, y_train, flagvec*t.from_numpy(lossweight), device2)
        floss = focal(out, y_train, flagvec*t.from_numpy(lossweight), device2)
        optimizer.zero_grad()
        (loss + (ceweight*celoss) + (focweight*floss)).backward()
        optimizer.step()
        train_loss += loss.item() + (ceweight*celoss).item() + (focweight*floss).item()

        y_train.detach_()
        out.detach_()
        del loss, x_train, y_train, out, floss, celoss
        
    print("train loss: ",train_loss/(len(traindataloader)-j),'\n')
    
    testtq = tqdm.tqdm(testdataloader)#, desc='loss', leave=True)
    test_loss = 0
    for x_test, y_test, flagvecTest in testtq:
        with t.no_grad():
            #SABOSNet_Model.eval()
            x_test = x_test.to(device1)
            y_test = y_test.to(device2)
            out = SABOSNet_Model(x_test)
            out = out.to(device2)
            loss = tversky_loss_wmask(out, y_test, flagvecTest*t.from_numpy(lossweight), device2)
            celoss = crossentropy(out, y_test, flagvecTest*t.from_numpy(lossweight), device2)
            floss = focal(out, y_test, flagvecTest*t.from_numpy(lossweight), device2)
            test_loss += loss.item() + (ceweight*celoss).item() + (focweight*floss).item()
            del out, x_test, y_test
            
    print(f"test loss: {test_loss/len(testdataloader)}")
    
    with open(file_name_1, "a") as f:
        f.write(f"epoch {epoch+1}")
        f.write(f"\ttrain loss: {train_loss/(len(traindataloader)-j)}")
        f.write(f"\ttest loss: {test_loss/len(testdataloader)}\n")
                        
    testloss = [0 for _ in range(7)]
    testtq = tqdm.tqdm(testdataloader, desc='loss', leave=True)
        
    for x_test, y_test, _ in testtq:
        with t.no_grad():
            #SABOSNet_Model.eval()
            x_test = x_test.to(device1)
            y_test = y_test.to(device2)
            o = SABOSNet_Model(x_test)
            o = o.to(device2)
            loss = caldice(o, y_test)
            testtq.set_description("test loss %f" % (sum(loss)/7))
            testtq.refresh() # to show immediately the update
            testloss = [l+tl if l != -1 else tl for l,tl in zip(loss, testloss)]  #testloss = [l+tl for l,tl in zip(loss, testloss)]
            del x_test, y_test, o
        
    testloss = [l / len(testtq) for l in testloss]
    with open(file_name_2, "a") as f:
        f.write(f"epoch {epoch+1}")
        f.write('\tDice coeff %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n' % tuple(testloss))
            
    if (epoch+1)%10==0:
        
        t.save({
        'epoch': epoch,
        'model_state_dict':  SABOSNet_Model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, epoch150_50_path_name)
        
        with open(file_name, "a") as f:
            f.write(f"epoch {epoch+1}")
            f.write('\tDice coeff %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n' % tuple(testloss))
        
        for x,y,_ in testdataloader:
            with t.no_grad():
                y_pred = SABOSNet_Model(x.to(device1))
                y_predDash = get_yPredDash(y_pred)
                np.save(y_pred_path,y_predDash)
                np.save(y_true_path,y.cpu())
                #print(y_pred.shape, y.shape) 
                break
