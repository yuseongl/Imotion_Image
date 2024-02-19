# Accuracy 값을 칼럼으로 갖는 데이터프레임 생성

import pandas as pd
import os ,json
import pandas as pd
from pathlib import Path


def make_df(accuracy, class_accuracy, df, name, path):

    try:
        # 주피터 노트북에서 실행하는 경우
        current_dir = os.getcwd()
        config_path = os.path.join(current_dir, 'config', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    except (FileExistsError, OSError):
        try:
            # main.py에서 실행하는 경우
            with open('Project_START/config/config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print("Error: Failed to load config file.")

    config_df = pd.DataFrame([config])
    accuracy_df = pd.DataFrame({"Accuracy": [accuracy]})
    dic = {}
    for i in class_accuracy:
        colums = i[0]
        value = i[1]
        dic[colums] = [value]
    # 데이터프레임 생성
    class_accuracy_df = pd.DataFrame(dic)

    list_data = df.values.tolist()

    cm_df = pd.DataFrame({"Confusion_matrix": [list_data]})



    result_df = pd.concat([accuracy_df, class_accuracy_df, cm_df, config_df], axis=1)

    try:
        # 주피터 노트북에서 실행하는 경우
        save_dir = Path('result/{}'.format(path))
        save_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(f'result/{path}/{name}_result.csv', index=False)
        print("Result saved successfully.")
        
    except FileNotFoundError:
        print("Error: Failed to save result.")
    
    return result_df

