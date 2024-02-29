import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from ultralytics import YOLO
import cv2
import google.generativeai as genai

import torchvision.transforms as transforms
from models.ResNet import resnet101

from diffusers import StableDiffusionPipeline
from translate import Translator

import os # 경로 탐색


def save_uploaded_file(directory, file):
    # 1. 저장할 디렉토리(폴더) 있는지 확인
    #   없다면 디렉토리를 먼저 만든다.
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    
    # 2. 디렉토리가 있으니, 파일 저장
    with open(os.path.join("", file.name), 'wb') as f:
        f.write(file.getbuffer())
    return st.success('파일 업로드 성공!')

def _default(file):
    if file != "":
        # YOLO 모델을 지정된 가중치 파일("best.pt")로 로드합니다
        model = YOLO("yolov8l-face.pt")  # COCO dataset으로 pretrained된 model을 불러옴
        # model = YOLO("emotion_dectect.pt")

        # 입력 이미지의 경로를 지정합니다
        img_path = file

        # YOLO를 사용하여 이미지에서 객체 감지를 실행합니다
        results = model(img_path)


        # 감지된 객체의 경계 상자를 가져옵니다
        boxes = results[0].boxes
        class_indices = results[0].names

        # 클래스 인덱스를 클래스 이름으로 변환하는 딕셔너리를 생성합니다
        index_to_class = {0:'angry',
                            1:'anxiety',
                            2:'embarrass',
                            3:'happy',
                            4:'normal',
                            5:'pain',
                            6:'sad'}

        # 입력 이미지를 읽어옵니다
        img = cv2.imread(img_path)


        # 감지된 객체 주위에 사각형을 그립니다
        for box, class_index in zip(boxes, class_indices):
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])

            # 원본 이미지에서 얼굴 부분을 잘라냅니다
            face = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            cv2.imwrite("testing.jpeg", face)

            return "testing.jpeg"


def _classification(file):
    from PIL import Image
    _default(file)

    file = Image.open(file)  # 이미지 경로 지정
    
    # 이미지 크기를 124x124로 조정
    transform = transforms.Compose([
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    file = transform(file)
    file = torch.unsqueeze(file, 0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    file = file.to(device)
                
    model = resnet101().to(device)
    checkpoint = torch.load('ResNet101best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(file.shape)
    model.eval()
    # 테스트 데이터로더에서 데이터를 가져와 모델에 통과시킵니다.
    with torch.no_grad():
        outputs = model(file)
    
                
    classes = ['angry','anxiety','embarrass','happy','normal','pain','sad']

    probabilities = torch.softmax(outputs, dim=1)
    predicted_class_index = torch.argmax(probabilities, dim=1)
    predicted_class_label = classes[predicted_class_index]

    return predicted_class_label


def _detection(file):
    if file != "":
        # YOLO 모델을 지정된 가중치 파일("best.pt")로 로드합니다
        model = YOLO("yolov8l-face.pt")  # COCO dataset으로 pretrained된 model을 불러옴
        # model = YOLO("emotion_dectect.pt")

        # 입력 이미지의 경로를 지정합니다
        img_path = file

        # YOLO를 사용하여 이미지에서 객체 감지를 실행합니다
        results = model(img_path)


        # 감지된 객체의 경계 상자를 가져옵니다
        boxes = results[0].boxes
        class_indices = results[0].names

        # 클래스 인덱스를 클래스 이름으로 변환하는 딕셔너리를 생성합니다
        index_to_class = {0:'angry',
                            1:'anxiety',
                            2:'embarrass',
                            3:'happy',
                            4:'normal',
                            5:'pain',
                            6:'sad'}

        # 입력 이미지를 읽어옵니다
        img = cv2.imread(img_path)


        # 감지된 객체 주위에 사각형을 그립니다
        for box, class_index in zip(boxes, class_indices):
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])

            # 이미지 위에 사각형을 그립니다
            cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (50, 200, 129), 2) #색상 값과 선 두께
            # (50, 200, 129):
            # BGR (파란색, 초록색, 빨간색) 색상 값
            # 2 # 선의 두께가 2픽셀임


            # 클래스 인덱스를 클래스 이름으로 변환합니다
            class_name = index_to_class[class_index]

            # 이미지 위에 클래스 이름을 그립니다
            cv2.putText(img, str(class_name), (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,200,129), 2)

            
            cv2.imwrite("testing.jpeg", img)

            return st.image("testing.jpeg")
    
        return st.success('파일 모델적용 성공!')
    

def click_button(img):
    if img != None:
        save_uploaded_file('image', img)
        st.image(img.name)
        result = _classification(img.name)
        st.write(result)

        return result

def imotion_text(img):
    if img != None:
        save_uploaded_file('image', img)
        result = _classification(img.name)
        st.write(result)

        return result

def _clear_session():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
    

def stabled_diffusion(text):
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"

    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                   torch_dtype=torch.float16).to('cuda')
    
    t = Translator(from_lang="ko", to_lang='en')
    t = t.translate(text)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(t)
    img = pipe(t).images[0]

    img.save('./test.jpg')
    

# 기본 형식
def main():
    # st.title('앱 데시보드')
  
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    menu = ['Object-Detection', 'Classification', 'Ent 추천', '그림 그리기', 'Testing']

    choice = st.sidebar.selectbox('메뉴', menu)
    
    detect = ['yolo-v8','resnet','vgg','vit','revit','cnn']
    classificate = ['CNN','VGG','ResNet','vit','revit']

    
    st.session_state.disabled = False
    if choice == menu[0]:
        _clear_session()
        img_file = None
        
        # st.sidebar.selectbox('모델', detect)
        st.subheader('Detection 이미지 파일 업로드')
        if st.radio(label = '이미지 등록 방법', options = ['카메라', '이미지 업로드']) == '카메라':
            camera_file = st.camera_input("Take a picture")
            file = camera_file
        else:
            img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
            file = img_file


        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) #radio 수평정렬
       
        if st.button("TEST 적용", use_container_width=True):
            if file != None or st.session_state != None:
                # st.write(imotion)
                save_uploaded_file('image', file)
                st.image(file.name)
                _detection(file.name)
                st.write('YOLO 적용 완료')
            else:
                st.error("이미지를 등록하세요.")              



    elif choice == menu[1]:
        _clear_session()
        img_file = None

        # st.sidebar.selectbox('모델', classificate)
        st.subheader('Classification 이미지 파일 업로드')
        if st.radio(label = '이미지 등록 방법', options = ['카메라', '이미지 업로드']) == '카메라':
            camera_file = st.camera_input("Take a picture")
            file = camera_file
        else:
            img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
            file = img_file
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) #radio 수평정렬

        if st.button("TEST 적용", use_container_width=True):
            if file != None:
                # st.write(imotion)
                imotion = click_button(file)
            else:
                st.error("이미지를 등록하세요.")

    elif choice == menu[2]:
        imotion = '' #gemini 오류 방지
        
        st.subheader('Classification 이미지 파일 업로드')
        if st.radio(label = '이미지 등록 방법', options = ['카메라', '이미지 업로드']) == '카메라':
            camera_file = st.camera_input("Take a picture")
            file = camera_file
        else:
            img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
            file = img_file
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) #radio 수평정렬
        
        recommend = st.radio(label = '추천', options = ['노래 추천', '영화 추천', "공감의 말"])
        
        if st.button("TEST 적용", use_container_width=True):
            if file != None:
                # st.write(imotion)
                imotion = click_button(file)
            else:
                st.error("이미지를 등록하세요.")

        
        
        # Google API key
        # 직접 Gemini api_key 입력하기
        st.session_state.api_key = ''
        
        if "api_key" not in st.session_state:
            try:
                st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
            except:
                st.session_state.api_key = ""
                st.write("Your Google API Key is not provided in `.streamlit/secrets.toml`, but you can input one in the sidebar for temporary use.")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Sidebar for parameters
        with st.sidebar:
        # Google API Key
            if not st.session_state.api_key:
                st.header("Google API Key")
                st.session_state.api_key = st.text_input("Google API Key", type="password")
            else:
                genai.configure(api_key=st.session_state.api_key)

            # ChatCompletion parameters
            # st.header("Parameters")
            model_name = st.selectbox("model_name",['gemini-pro'])
            
            generation_config = {
                "temperature": 0.9,
                "max_output_tokens": 2048,
                "top_k": 40,
                "top_p": 0.95,
            }

        # Display messages in history
        for msg in st.session_state.messages:
            if parts := msg.parts:
                with st.chat_message('human' if msg.role == 'user' else 'ai'):
                    for p in parts:
                        st.write(p.text)

        # '''
        
        # Chat input
        if imotion != '':
            prompt = "I feel " + imotion + ", 기분에 맞는 " + recommend +" 10개 해줘 한국어로"
            if recommend == '공감의 말':
                # prompt = "I feel " + imotion + ', 내 기분에 맞는 공감의 말 한문장만 한국말로 해줘'
                if imotion == 'happy':
                    imotion = '나 행복해 내 말 좀 들어줘'
                elif imotion == 'sad':
                    imotion = '나 슬퍼 내 말 좀 들어줘'
                elif imotion == 'anxiety':
                    imotion = '나 너무 긴장돼 내 말 좀 들어줘'
                elif imotion == 'embarrass':
                    imotion = '나 너무 황당해 내 말 좀 들어줘'
                elif imotion == 'normal':
                    imotion = '나 기분이 그냥그래 내 말 좀 들어줘'
                elif imotion == 'pain':
                    imotion = '나 아파 내 말 좀 들어줘'
                else:
                    imotion = ''
                
                
                prompt = imotion

            # Generate
            model = genai.GenerativeModel(model_name=model_name,
                                                generation_config=generation_config)
            chat = model.start_chat(history=st.session_state.messages)
                
            response = chat.send_message(prompt, stream=True)
            

            # Stream display
            with st.chat_message("ai"):
                placeholder = st.empty()
            
                if recommend != '공감의 말':
                    text = imotion + ' 기분에 맞는 '+ recommend +'\n'
                else:
                    text = imotion + '\n'
            
            # Stream display
            for chunk in response:
                text += chunk.text
                placeholder.write(text + "▌")
            placeholder.write(text)

        # Chat input
        if prompt := st.chat_input("What is up?"):
            # Display user message
            with st.chat_message('human'):
                st.write(prompt)

            # Generate
            model = genai.GenerativeModel(model_name=model_name,
                                            generation_config=generation_config)
            chat = model.start_chat(history=st.session_state.messages)
            response = chat.send_message(prompt, stream=True)


            # Stream display
            with st.chat_message("ai"):
                placeholder = st.empty()
            text = ''
            for chunk in response:
                text += chunk.text
                placeholder.write(text + "▌")
            placeholder.write(text)

            st.session_state.messages = chat.history
            # ''' 
            _clear_session()
            
        


            


    elif choice == menu[3]:
        _clear_session()
        prompt = st.text_input("그리고 싶은 그림을 입력하세요")
        st.write(prompt)

        if st.radio(label = '이미지 등록 방법', options = ['카메라', '이미지 업로드']) == '카메라':
            camera_file = st.camera_input("Take a picture")
            file = camera_file
        else:
            img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])            
            file = img_file
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True) #radio 수평정렬


        if st.button("TEST 적용", use_container_width=True):
            if file != None:
                imotion = imotion_text(file)
                text = imotion + "한 " + prompt
                stabled_diffusion(text)
                st.image('test.jpg')

            else:
                st.error("이미지를 등록하세요.")



        

if __name__ == '__main__':
    main()