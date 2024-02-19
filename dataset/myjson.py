import json
import pandas as pd

def myjson():  
    x = ["ang", "anx","emb", "happy","normal","pain","sad"]
    # JSON 파일 읽기
    rows = []
    for i in x:
        with open(f'라벨링데이터/EMOIMG_{i}_SAMPLE/img_emotion_sample_data({i}).json', 'r') as file:
            data = json.load(file)
    ##########################
        for item in data:
            row = {
                'filename': item['filename'],
                'gender': item['gender'],
                'age': item['age'],
                'isProf': item['isProf'],
                'faceExp_uploader': item['faceExp_uploader'],
                'bg_uploader': item['bg_uploader'],
                'annot_A_maxX': item['annot_A']['boxes']['maxX'],
                'annot_A_maxY': item['annot_A']['boxes']['maxY'],
                'annot_A_minX': item['annot_A']['boxes']['minX'],
                'annot_A_minY': item['annot_A']['boxes']['minY'],
                'annot_A_faceExp': item['annot_A']['faceExp'],
                'annot_A_bg': item['annot_A']['bg'],
                'annot_B_maxX': item['annot_B']['boxes']['maxX'],
                'annot_B_maxY': item['annot_B']['boxes']['maxY'],
                'annot_B_minX': item['annot_B']['boxes']['minX'],
                'annot_B_minY': item['annot_B']['boxes']['minY'],
                'annot_B_faceExp': item['annot_B']['faceExp'],
                'annot_B_bg': item['annot_B']['bg'],
                'annot_C_maxX': item['annot_C']['boxes']['maxX'],
                'annot_C_maxY': item['annot_C']['boxes']['maxY'],
                'annot_C_minX': item['annot_C']['boxes']['minX'],
                'annot_C_minY': item['annot_C']['boxes']['minY'],
                'annot_C_faceExp': item['annot_C']['faceExp'],
                'annot_C_bg': item['annot_C']['bg']
            }
            rows.append(row)
    # 데이터프레임 생성
    df = pd.DataFrame(rows)
    # 데이터프레임 확인
    return df
