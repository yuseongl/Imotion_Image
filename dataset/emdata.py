from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
import torchvision
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def emdata(batch_size = 4, size = 224):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((size,size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    batch_size = batch_size
    # dataset = torchvision.datasets.ImageFolder(root = './data/train_crop/image', transform=transform)
    dataset = torchvision.datasets.ImageFolder(root = '/home/KDT-admin/14000_crop_landmark', transform=transform)
    # 데이터를 훈련 세트와 테스트 세트로 무작위로 나누기

    # 이미지 데이터와 레이블을 numpy 배열로 변환
    X = np.array([sample[0] for sample in dataset.samples])  # 이미지 데이터
    y = np.array([sample[1] for sample in dataset.samples])  # 레이블

    # 전체 데이터를 train+val 세트와 test 세트로 나눕니다.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=777)  # test_size를 0.1로 설정
    train_idx, val_idx = next(sss.split(X, y))


    # 훈련 세트와 검증 세트를 Subset으로 생성
    # 분할된 인덱스를 기반으로 데이터셋 생성
    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)

    # DataLoader 생성
    trainloader = DataLoader(train_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                            num_workers = 2)
    valloader = DataLoader(val_data, 
                           batch_size=batch_size, 
                           shuffle=True,
                            num_workers = 2)

    return trainloader,valloader

def emdata_tst(batch_size = 4, size = 224):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((size,size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    batch_size = batch_size
    # dataset = torchvision.datasets.ImageFolder(root = './data/test/image', transform=transform)
    dataset = torchvision.datasets.ImageFolder(root = '/home/KDT-admin/test_set1000/image', transform=transform)
    # 데이터를 훈련 세트와 테스트 세트로 무작위로 나누기

    # DataLoader 생성
    tstloader = DataLoader(dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers = 2)

    return tstloader