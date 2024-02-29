from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Whale/3.22.205.26 Safari/537.36'}

song_list = []
song_text=[]
title = []
singer = []
txt = []
img = []
like = []

def init_selenium():
    """
    해당 함수는 원활한 Selenium 사용을 위해 설정을 진행하는 함수입니다.
    첨부된 주석을 통해 해당 코드의 기능을 확인할 수 있습니다.
    """
    driver_options = Options()
    # 하기 코드는 False로 설정시 자동으로 웹 브라우저가 종료될 수 있게 설정합니다.
    driver_options.add_experimental_option("detach", True) 
    # 하기 코드는 불필요한 경고 메시지가 출력되지 않게 설정합니다.
    driver_options.add_experimental_option("excludeSwitches", ["enable-logging"]) 
    # 하기 코드는 WebDriver를 자동으로 관리하는 모듈을 호출하여 설치합니다.
    auto_driver = Service(ChromeDriverManager().install())

    # 설정과 WebDriver를 이용해 크롤러를 driver 변수에 선언합니다.
    driver = webdriver.Chrome(service = auto_driver, options = driver_options)
    return driver

def crawling_info():
    driver = init_selenium()
    url = "https://www.melon.com/genre/song_list.htm?gnrCode=GN0200&steadyYn=Y"

    driver.get(url)
    wait = WebDriverWait(driver, 10)

    # 테이블의 각 행을 가져와서 딕셔너리로 저장
    song_info = []
    table_rows = driver.find_elements(By.CSS_SELECTOR, '#frm > div > table > tbody > tr')

    for row in table_rows:
        checkbox_value = row.find_element(By.CSS_SELECTOR, 'td input.input_check').get_attribute('value')
        image_link = row.find_element(By.CSS_SELECTOR, 'td div.wrap a.image_typeAll img').get_attribute('src')

        song_info.append({
            'id': checkbox_value,
            'img': image_link,
            'text': row.text
        })

    for song in song_info:
        song_l = song['text'].split('\n')
        title.append(song_l[1])
        singer.append(song_l[2])
        txt.append(song['id'])
        img.append(song['img'])
        like.append(song_l[5])
        
    # 노래명, 가수명, 노래 가사, 앨범사진, 좋아요 수

    # 브라우저 닫기
    driver.quit()
    
def crawling_text():
    
    init_selenium()
    
    # Chrome 드라이버 서비스를 설정
    driver = init_selenium()
    
    for id in txt:
        # 크롤링할 페이지가 있는 주소값을 넣기
        url = f"https://www.melon.com/song/detail.htm?songId={id}"

        driver.get(url)
        wait = WebDriverWait(driver, 10)

        # lyric 클래스를 갖는 요소를 찾아서 텍스트를 출력
        try:
            lyric_text = driver.find_element(By.XPATH, '//*[@id="d_video_summary"]').text
            song_text.append(lyric_text)
        except NoSuchElementException as e:
            print("Element not found:", e)

    # 브라우저 닫기
    driver.quit()
    
    for ss in zip(title, singer, song_text, img,like):
        song_list.append(ss)
        
    print(len(title),len(singer), len(song_text),len(img),len(like))
    
    df = pd.DataFrame(
        {
        "title" :title,
        "singer" :singer, 
        "song_text": song_text,
        "img": img,
        "like" : like
        }
    )
    
    df.to_csv('hippop.csv', index=False)