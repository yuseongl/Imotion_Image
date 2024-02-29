import pandas as pd
from MusicRecommend import find_similar_songs

import warnings
warnings.filterwarnings('ignore')

songs_data = pd.read_csv('song_emotion.csv') # title, genre, singer, img를 컬럼으로 가지는 csv형태의 파일

recomend = find_similar_songs('오늘 친구들이랑 꿔바로우를 먹으러 갔는데, 손님들이 너무 많은 거야~\n그래서 1시간을 기다려서 들어갔다? 근데 사장님이 너무 불친절한 거 있지?\n 그리고 주문한 음식도 1시간이나 늦게 나와서 기분이 너무 별로였어!',songs_data)
print(recomend)
