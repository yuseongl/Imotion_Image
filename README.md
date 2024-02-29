# Imotion_Image


## 프로젝트 정보
### 주제
얼굴을 인식하여 감정 분류를 해서 다양한 기능을 구현하는 프로젝트

### 선정 배경
고객들의 감정 상태를 실시간으로 이해하고 분석하는 것은 현대 사회에서 중요한 과제 중 하나입니다. 특히, 디지털 커뮤니케이션의 증가와 함께 감정적인 상태를 정확하게 파악하여 적절한 서비스를 제공하는 것이 필요합니다. 이러한 필요성을 인지하여, 우리는 감정 분류 서비스를 기획하게 되었습니다.

### 기획 의도
기본적인 감정분류를 통해 여러가지 TASK를 진행해보며 다양한 기능을 체험할 수 있는 서비스를 기획

### 프로젝트 기간
2024.02.13 - 2024.02.29
### 개발자
```조장``` 정서익

```조원``` 이유성, 윤재현, 전정현

### 개발 환경
|IDE|GPU 서버|프로그래밍 언어|
|:-----:|:-----:|:-----:|
|<img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"><br/><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"><br/><img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">|Microsoft Azure A100 GPU|<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">|
---- -
### Dataset
[<a href="https://www.aihub.or.kr/">AI 허브</a>] 한국인 감정인식을 위한 복합 영상

<div>
구축 목적 - 한국인의 얼굴 표정과 장소 맥락을 고려하여 인공지능이 사람의 감정을 이해할 수 있는 학습 모델 개발<br/>
활용 분야 - 감정분석을 통한 소비자의 만족도 측정, 감정기록을 통한 멘탈 헬스케어, 감정인식을 통한 감성 컴퓨팅(Affective Computing) 분야 등<br/><br/>
라벨링 된 감정 이미지 데이터(기쁨, 당황, 분노, 상처, 슬픔, 중립)와 해당 감정의 Json파일로 각각 매핑
</div>

### 데이터 경로
```
# 데이터 호출 경로
# dataset/emdata.py 에서 수정 가능
```

```
train dataset
data/train/image/감정 - 감정별 디렉터리명 안에 이미지 존재
data/train/label/ - 감정별 라벨링된 JSON 파일
```
```
test dataset
test_set1000/image/감정 - 감정별 디렉터리명 안에 이미지 존재
test_set1000/label - 감정별 라벨링된 JSON 파일
```


### DataSet
![image](https://github.com/lily39910/Imotion_Image/assets/92513469/cef83e1f-4823-4fc2-9296-1a0a2941b462)


### EDA
- YOLOv8 모델 중 얼굴 탐지만 전문적으로 학습한 모델을 활용하여 얼굴을 crop
- 링크(https://github.com/akanametov/yolov8-face)
- **해당 과정 모식도**
![image](https://github.com/lily39910/Imotion_Image/assets/92513469/29322451-72cb-4464-b980-06333d840e40)


- dlib 활용하여 얼굴 랜드마크 추가 진행
- **해당 과정 모식도**
![image](https://github.com/lily39910/Imotion_Image/assets/92513469/64537c48-059c-4344-b55a-a78a0df2f5e5)




## 시작 가이드
### Installation
```
$ git clone https://github.com/lily39910/Imotion_Image.git
$ cd Imotion_Image
$ pip install -r requirements.txt
```
### trian model & testing model file
```
# config/config.json에서 학습시킬 모델의 종류와 하이퍼파라미터 및 스케줄러 적용
# 모델의 종류에 대한 정보는 models/model_selection.py에서 확인 가능
$ python main.py

# output_model에 pth 파일 생성되고, result 하위 폴더로 모델명 폴더가 생성되며 그 안에 학습한 모델의 accuracy, confusion_matrix, F1, Precision, Recall, result.csv, train_loss, val_loss 파일들이 생성된다

# 기존 학습한 모델을 이어서 학습하고 싶으면 add_train.py로 config 설정을 동일하게 지정하면 이어서 학습 가능하다. 
$ python add_train.py

# 학습한 모델의 성능을 검증하기 위해 test.py를 실행해서 확인 가능하다
$ python test.py
```

### 시각적으로 모델을 적용한 기능 활용
```
# streamlit을 이용하여 Web에서 모델을 적용한 기능들을 확인 가능하다.
$ streamlit run stream.py
```

```
# stream.py
# Google API key
# 직접 Gemini api_key 입력하기(298Line)
st.session_state.api_key = ''
```
