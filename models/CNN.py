import torch.nn.functional as F
import torch
import torch.nn as nn

#CNN base 성능 낮음---> VGG, ResNet, VIT 비교해보고
#가중치 가져와서 다시 간단하게 학습하고 돌려보기,
#모델가져와서 네트워크 바꿀 수 있음, 레이어마다
# 수정 가능함
# 레이어 수정 가능함, --> 성능이 좋다고 보장은 안된다. 원래 기존 레이어를 건드리는게 좋다.
## 파인튜닝시 기존 레이어는 안건드린다 성능적인 면에서 떨어질 수 있음
# 에폭 낮추고 파인튜닝함


class CNN_128(torch.nn.Module):
    def __init__(self):
        super(CNN_128, self).__init__()
        # Torch tensor dim. (bath_size, C, H, W)
        # L1 ImgIn shape=(batch_size, 3,  128, 128)
        #    Conv     -> (batch_size, 32, 128, 128)
        #    Relu     -> (batch_size, 32, 128, 128)
        #    Pool     -> (batch_size, 32, 64, 64)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # L2 ImgIn shape=(batch_size, 32, 64, 64)
        #    Conv      ->(batch_size, 64, 64, 64)
        #    Relu      ->(batch_size, 64, 64, 64)
        #    Pool      ->(batch_size, 64, 32, 32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Final FC 8x8x64 inputs -> 4 outputs
        # resize 사진 전처리 크기를 바꾸면 이 파트를 꼭 바꿔야 함 27라인 
        self.fc1= torch.nn.Linear(32 * 32 * 64, 128, bias=True)
        self.fc2= torch.nn.Linear(128, 64, bias=True)
        self.fc3= torch.nn.Linear(64, 7, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        # dim : (batch_size, 64, 32, 32)
        out = out.view(out.size(0), -1)   
        # Flatten them for FC --> dim : (batch_size, 32*32*64)
        out = F.relu(self.fc1(out)) # dim : (batch_size, 4)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    

class CNN_224(torch.nn.Module):
    def __init__(self):
        super(CNN_224, self).__init__()
        # Torch tensor dim. (bath_size, C, H, W)
        # L1 ImgIn shape=(batch_size, 3,  224, 224)
        #    Conv     -> (batch_size, 32, 224, 224)
        #    Relu     -> (batch_size, 32, 224, 224)
        #    Pool     -> (batch_size, 32, 112, 112)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # L2 ImgIn shape=(batch_size, 32, 112, 112)
        #    Conv      ->(batch_size, 64, 112, 112)
        #    Relu      ->(batch_size, 64, 112, 112)
        #    Pool      ->(batch_size, 64, 56, 56)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Final FC 56x56x64 inputs -> 4 outputs
        # resize 사진 전처리 크기를 바꾸면 이 파트를 꼭 바꿔야 함 27라인 
        self.fc1= torch.nn.Linear(56 * 56 * 64, 512, bias=True)
        self.fc2= torch.nn.Linear(128, 64, bias=True)
        self.fc3= torch.nn.Linear(64, 7, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        # dim : (batch_size, 64, 56, 56)
        out = out.view(out.size(0), -1)   
        # Flatten them for FC --> dim : (batch_size, 56*56*64)
        out = F.relu(self.fc1(out)) # dim : (batch_size, 4)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out