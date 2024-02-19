import json
import torch
import time
from tqdm import tqdm
from utils.visual import Accuracy_CM_V
from eval import evaluate
from utils.dataframe import make_df
from models.model_selection import get_network

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'avaiable device : %s' % device)
# for reproducibility
from dataset.emdata import emdata_tst

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(777)
# if device == 'cuda':
#     torch.cuda.manual_seed_all(777)


def load_config(config_file):
    script_dir = os.getcwd()  # 현재 스크립트 파일의 디렉토리 경로
    config_path = os.path.join(script_dir, config_file)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 데이터셋 로드 및 전처리
    # classes = ['angry','anxiety','embarrass','happy','normal','pain','sad']
       
    #testloader = emdata_tst(config['batch_size'],config['resize']) # 8:1:1
    testloader, v = emdata_tst(config['batch_size'],config['resize']) # 8:1:1
    #1500x 1000 ~ 1480x1320 --> resize # 바꿔봐야 하는 거 (하이퍼 파라미터)
    #def emdata(batch_size = 32, size = 128): size가 resize를 의미함

    model = get_network(config["model_type"]).to(device)
    print(model)
    model.load_state_dict(torch.load('output_model/'+config["model_type"] + "_last_model.pth"))

    print("test data validation Start!")

#############################################################

    print('Finished Training')

    # 학습된 모델 저장
    df, fig, accuracy , class_acc = Accuracy_CM_V(model, v, config['name'], config['model_type'])

    # accuracy = acc(model, testloader)
    # print("---------------------------")
    # cacc = class_accuracy(model, testloader, classes)
    # print("---------------------------")
    # df, fig = cm_visual(model, testloader)
    # # plt.show()
    make_df(accuracy, class_acc ,df,config['name'], config['model_type'])

if __name__ == "__main__":
    
    config_file = "config/config.json"
    config = load_config(config_file)
    main(config)
