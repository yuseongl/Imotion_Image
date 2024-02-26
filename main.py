import json
import torch
import time
from tqdm import tqdm
from utils.visual import trnV_loss, accuracyV, Accuracy_CM_V, prf1V, top3_accuracyV
from utils.earlystop import EarlyStopper
from train import train
from eval import evaluate
from utils.dataframe import make_df
from models.model_selection import get_network
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import os

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print(f'avaiable device : %s' % device)
# for reproducibility
from dataset.emdata import emdata, emdata_tst

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(777)
if device == 'cuda:2':
    torch.cuda.manual_seed_all(777)


def load_config(config_file):
    script_dir = os.getcwd()  # 현재 스크립트 파일의 디렉토리 경로
    config_path = os.path.join(script_dir, config_file)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main(config):
    # 데이터셋 로드 및 전처리
    classes = ['angry','anxiety','embarrass','happy','normal','pain','sad']
       
    trainloader, valloader = emdata(config['batch_size'],config['resize']) # 8:1:1
    # tstloader = emdata_tst(config['batch_size'],config['resize'])
    #1500x 1000 ~ 1480x1320 --> resize # 바꿔봐야 하는 거 (하이퍼 파라미터)
    #def emdata(batch_size = 32, size = 128): size가 resize를 의미함

    model = get_network(config["model_type"]).to(device)
    print(model)

    # model.load_state_dict(torch.load('model.pth'))

    criterion = torch.nn.CrossEntropyLoss().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, config['scheduler']["T_0"], 
                                            config['scheduler']["T_mult"], 
                                            config['scheduler']["eta_min"])
    
    history = {
        "model_state_dict":model.state_dict(),
        "f1s":[],
        "precisions":[],
        "recalls":[],
        "c_f1s":[],
        "c_precisions":[],
        "c_recalls":[],
        "accuracies_top3":[],
        "train_losses":[],  # train loss 시각화 위한 리스트
        "val_losses":[],    # val loss 시각화 위한 리스트
        "accuracies":[],    # 에포크별 정확도를 저장할 리스트 초기화
        "lr":[]             # learning rate 를 저장할 리스트
    }

    start_time = time.time()
    early_stopper = EarlyStopper(config['patiece'] , 0)
    
    print("Training Start!")

#############################################################
    pbar = tqdm(range(config["epochs"]))
    for _ in pbar:  
        train_loss = train(model,criterion,optimizer,trainloader,device)
        history["train_losses"].append(train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        if config['scheduler']["mode"]:
            scheduler.step(train_loss)
        loss_val, accuracy, all_labels, all_predictions, accuracy_top5 = evaluate(model, criterion, valloader, device)
        history["val_losses"].append(loss_val)
        history["accuracies"].append(accuracy)
        history["accuracies_top3"].append(accuracy_top5)
        pbar.set_postfix(trn_loss=train_loss,
                         val_loss=loss_val,
                         accuracy=accuracy)
        early_stopper.early_stop(model, loss_val, config["model_type"]+'best_model.pth')
        
        #######################################################
        precision = precision_score(all_labels, all_predictions , average=None,zero_division=1)
        recall = recall_score(all_labels, all_predictions , average=None,zero_division=1)
        f1 = f1_score(all_labels, all_predictions , average=None,zero_division=1)
        history["c_precisions"].append(precision)
        history["c_recalls"].append(recall)
        history["c_f1s"].append(f1)
        ####################################average = None 각 클래스별로 구해서 배열로 넘겨줌
        precision = precision_score(all_labels, all_predictions , average='macro')
        recall = recall_score(all_labels, all_predictions , average='macro')
        f1 = f1_score(all_labels, all_predictions , average='macro')
        history["precisions"].append(precision)
        history["recalls"].append(recall)
        history["f1s"].append(f1)
        
        
######################################################
    #print('[Epoch: {:>4}] Train Loss = {:.9f}, Val Loss = {:.9f}, val Accuracy = {:.2f}%'.format(epoch + 1, train_loss, loss_val, accuracy))
##########################################################################

    print(time.time()-start_time)
    print('Finished Training')

    # 학습된 모델 저장
    torch.save(history, 'output_model/'+config["model_type"] + '_' + config['name'] +"_last_model.pth")

    accuracyV(history["accuracies"], config['name'], config['model_type'])
    trnV_loss(history["train_losses"],history["val_losses"], 
              config['name'], config['model_type'])
    top3_accuracyV(history["accuracies_top3"], config['name'], config['model_type'])
    prf1V(history["f1s"],history["precisions"],history["recalls"],
          history["c_f1s"],history["c_precisions"],history["c_recalls"],
          config['name'], config['model_type'])
    df, fig, accuracy , class_acc = Accuracy_CM_V(model, valloader, 
                                                  config['name'], config['model_type'])

    # accuracy = acc(model, testloader)
    # print("---------------------------")
    # cacc = class_accuracy(model, testloader, classes)
    # print("---------------------------")
    # df, fig = cm_visual(model, testloader)
    # # plt.show()
    make_df(accuracy, class_acc ,df, config['name'], config['model_type'])

if __name__ == "__main__":
    
    config_file = "config/config.json"
    config = load_config(config_file)
    main(config)
