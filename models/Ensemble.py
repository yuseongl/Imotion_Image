from torchvision.models import resnet101, ResNet101_Weights, vit_b_16, ViT_B_16_Weights
import torch
import torch.nn as nn
# from models.CNN import CNN_128,CNN_224

class Ensemble(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_dim):
        super().__init__()
        self.resnet = PreTrainResNet(input_dim, output_dim, mlp_dim)
        self.vit = PreTrainVit(input_dim, output_dim, mlp_dim)

        self.ensemble_layer = nn.Sequential(
            torch.nn.Linear(in_features=output_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=output_dim)
        )
        
        self.res_layer = nn.Sequential(
            torch.nn.Linear(in_features=output_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=output_dim)
            )
        
        self.fc_layer = nn.Sequential(
            torch.nn.Linear(in_features=output_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=output_dim)
            )
        
        
    def forward(self,x):
        res_skip = self.resnet(x)
        vit_skip = self.vit(x)
        skip = res_skip + vit_skip
        ensemble = nn.Parameter(res_skip) + nn.Parameter(vit_skip)
        skip2 = self.ensemble_layer(ensemble+skip)
        skip3 = self.res_layer(ensemble+skip+skip2)
        x = self.fc_layer(ensemble+skip+skip2+skip3)
        
        return x
    
    
class PreTrainResNet(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_dim):
        super().__init__()
        self.model = resnet101(weights= ResNet101_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.model.fc = torch.nn.Linear(in_features=512*4, out_features=input_dim)
        
        self.layer = nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=input_dim)
            )
        
        self.res_layer = nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=input_dim)
            )
        self.fc_layer = nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=output_dim)
            )
        
    def forward(self,x):
        x = self.model(x)
        skip = x
        x = self.layer(x)
        x = self.res_layer(skip+x)
        skip2 = x
        x = self.fc_layer(skip+skip2+x)
        return x
    
class PreTrainVit(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_dim):
        super().__init__()
        self.model = vit_b_16(weights= ViT_B_16_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.model.heads = torch.nn.Linear(in_features=768, out_features=input_dim)
        
        self.layer = nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=input_dim)
            )
        
        self.res_layer = nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=input_dim)
            )
        self.fc_layer = nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=mlp_dim),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=mlp_dim, out_features=output_dim)
            )
        
    def forward(self,x):
        x = self.model(x)
        skip = x
        x = self.layer(x)
        x = self.res_layer(skip+x)
        skip2 = x
        x = self.fc_layer(skip+skip2+x)
        return x