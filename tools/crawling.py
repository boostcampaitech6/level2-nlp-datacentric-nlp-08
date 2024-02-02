import re
from tqdm.auto import tqdm

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait   
from webdriver_manager.chrome import ChromeDriverManager
    
class Crawl_news:
    def __init__(self, sid1):
        self.targets = {100: 6, 101: 1, 102: 2, 103: 3, 104: 4, 105: 0}
        self.columns = ["text","target","content","url"]
        self.sid1 = sid1
        
    def create_driver(self):
        chromedriver_path = '/data/ephemeral/level2-nlp-datacentric-nlp-08/chromedriver-linux64/chromedriver'
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        service = Service(executable_path=chromedriver_path)
        return webdriver.Chrome(service=service, options=options)
        
        
    def clean_sentence(self, sentence):
        edited_sentence = re.sub(r'"', ' ', sentence)
        edited_sentence = re.sub(r'\[[^\]]+\]',' ',edited_sentence)
        edited_sentence = re.sub(r'[^\w\sㄱ-힣一-龠ぁ-んァ-ン!@#$%^&*()_+\-=\[\]{}|;:\'<>,.?/\\~]', '', edited_sentence)  
        edited_sentence = edited_sentence.strip()
        return edited_sentence
        
    def get_articles(self, target_number):
        base_url = f'https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1={self.sid1}'
        
        main_driver = self.create_driver()
        article_driver = self.create_driver()
        
        main_driver.get(base_url)
        
        page = 1
        reach_to_target = False
        total_articles = 0
        while not reach_to_target:
            WebDriverWait(main_driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '._paging'))
            )       
            
            if page != 1:
                try:
                    button_xpath = f'//*[@id="paging"]/a[contains(@href, "page={page}")]'

                    page_button = main_driver.find_element(By.XPATH, button_xpath)
                    page_button.click()
                    print(f"현재 페이지: {page}")
                except:
                    continue
            
            WebDriverWait(main_driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.photo'))
            )
            
            article_urls = []
            links = main_driver.find_elements(By.CSS_SELECTOR, 'div.section_body dt.photo a')
            for link in links:
                try:
                    url = link.get_attribute('href')
                    article_urls.append(url)
                except:
                    continue
                    
            num_articles = self.extract_content(article_urls,article_driver)
            total_articles += num_articles
            
            if total_articles > target_number:
                reach_to_target = True
            else:
                print(f"추출한 기사 수: {total_articles}")
                page += 1
        
    def extract_content(self, article_urls, article_driver):
        articles = pd.DataFrame(columns=self.columns)
        
        for url in tqdm(article_urls, desc="extracting articles"):
            article_driver.get(url)
            WebDriverWait(article_driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.c_text'))
            )

            copyright_element = article_driver.find_element(By.CSS_SELECTOR, ".c_text")
            if "AI" in copyright_element.text:
                continue
            
            title_element = article_driver.find_element(By.CSS_SELECTOR, '.media_end_head_headline')
            title = title_element.text.strip()
            
            content_element = article_driver.find_element(By.ID, 'dic_area')
            content_text = content_element.text
            try:
                content = content_text.split('\n\n')[2]
            except:
                content = content_text[:100]
            content = content.replace("\n", ' ')
            
            title = self.clean_sentence(title)
            content = self.clean_sentence(content)
            
            new_row = pd.DataFrame({
                "text": [title],
                "target": [self.targets[self.sid1]],
                "content": [content],
                "url": [url]
            })
            
            articles = pd.concat([articles, new_row], ignore_index=True)
        num_articles = len(articles)
        articles.to_csv("./crawling_data_v2.csv", mode="a", sep=",", index=False, header=False)  
        
        return num_articles  
            
    

if __name__ == '__main__':
    subject_label = {
        '정치': 100,
        '경제': 101,
        '사회': 102,
        '문화생활': 103,
        '세계': 104,
        'IT과학': 105
    }
    success = False
    while not success:
        print(f'--- 원하는 분야를 입력해주세요 {list(subject_label.keys())} ---')
        try:
            subject = input()
            target_label = subject_label[subject]
            success = True
        except KeyboardInterrupt:
            print('코드 실행을 중단합니다.')
            exit(1)
        except:
            print('잘못된 분야입니다.')
    success = False
    while not success:
        print('--- 크롤링 할 기사의 개수를 입력하세요 ---')
        try:
            target_number = int(input())
            success = True
        except KeyboardInterrupt:
            print('코드 실행을 중단합니다.')
            exit(1)
        except:
            print('정수만 입력할 수 있습니다.')
            
    crawler =  Crawl_news(sid1=target_label)
    crawler.get_articles(target_number)
      