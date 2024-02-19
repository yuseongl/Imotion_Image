from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
# 모델을 평가 모드로 설정


def trnV_loss(train_losses, val_losses, name, path):
    # Train Loss 그래프
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, len(train_losses)), train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Epochs')
    plt.legend()
    plt.show()
    

    # 그래프 저장을 시도하고, 실패할 경우 대체 경로를 시도함
    try:
        # 주피터 노트북에서 실행하는 경우
        save_dir = Path('result/{}'.format(path))
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'result/{path}/{name}_train_loss.png')
        print("Train and validation loss graph saved successfully.")
        
    except FileNotFoundError:
        print("Error: Failed to save train and validation loss graph.")

    # Val Loss 그래프
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, len(val_losses)), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

    # 그래프 저장을 시도하고, 실패할 경우 대체 경로를 시도함
    try:
        # 주피터 노트북에서 실행하는 경우
        save_dir = Path('result/{}'.format(path))
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'result/{path}/{name}_val_loss.png')
        print("Train and validation loss graph saved successfully.")

    except FileNotFoundError:
        print("Error: Failed to save train and validation loss graph.")


def accuracyV(accuracies, name, path):
    
    # Accuracy 그래프
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(accuracies)), accuracies, label='Accuracy')  # accuracies 리스트가 필요합니다.
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.show()

        # 그래프 저장을 시도하고, 실패할 경우 대체 경로를 시도함
    try:
        # 주피터 노트북에서 실행하는 경우
        save_dir = Path('result/{}'.format(path))
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'result/{path}/{name}_accuracy.png')
        print("Accuracy graph saved successfully.")

    except FileNotFoundError:
        print("Error: Failed to save accuracy graph.")

            

def Accuracy_CM_V(model, loader, name, path):
    # 모든 예측과 레이블을 저장할 리스트 초기화
    classes = ['angry','anxiety','embarrass','happy','normal','pain','sad']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_preds = []
    all_labels = []
    ####### 혼동행렬을 위한 리스트

    correct = 0
    total = 0
    ####### 정확도 측정을 위한 변수

    class_correct = list(0. for i in(range(7)))
    class_total = list(0. for i in range(7))
    ####### 클래스별 정확도 위한 변수

    model.eval()
    # 테스트 데이터로더에서 데이터를 가져와 모델에 통과시킵니다.
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            ###########################
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            #################위는 혼동행렬 아래 두줄은 정확도 측정
            total += labels.size(0) # the number of test image
            correct += (predicted == labels).sum().item()

            ###############클래스 정확도 측정위한 코드
            # c = (predicted == labels).squeeze()
            # for i in range(4):
            #     label = labels[i]
            #     class_correct[label] += c[i].item()
            #     class_total[label] += 1
            c = (predicted == labels)
            for i, (label, correct_prediction) in enumerate(zip(labels, c)):
                class_correct[label] += correct_prediction.item()
                class_total[label] += 1


    # 정확도 계산합니다.
    accuracy = 100 * correct//total    
    print(f'Accuracy of the network on the test images:{accuracy}%')

    #클래스별 정확도 계산합니다.
    result = []
    for i in range(7):
        accuracy = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (
            classes[i], accuracy))
        result.append([classes[i], accuracy])



    # 혼동 행렬을 계산합니다.
    cm = confusion_matrix(all_labels, all_preds)
    
    # 혼동 행렬을 데이터프레임으로 변환합니다.
    cm_df = pd.DataFrame(cm, index=[i for i in range(7)], columns=[i for i in range(7)])

    print(cm_df) # 혼동 행렬을 출력합니다.

    # 혼동 행렬 시각화로 넘겨주기 위한 코드 부분
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 10})  # 숫자 크기 조절
    # sns.heatmap(cm_df, annot=True, fmt='g', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


    # 결과 이미지 저장을 시도하고, 실패할 경우 대체 경로를 시도함
    try:
        # 주피터 노트북에서 실행하는 경우
        save_dir = Path('result/{}'.format(path))
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'result/{path}/{name}_confusion_matrix.png')
        print("Confusion matrix image saved successfully.")
    except FileNotFoundError:
        print("Error: Failed to save confusion matrix image.")

    return cm_df, fig ,accuracy, result
