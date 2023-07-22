from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service

import requests
import re
import os
import urllib.request
import time
import ssl
from tqdm import tqdm
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

os.chdir('./crawling/crawling_data2')

def get_category_name(category_a):
    category = category_a.get_attribute('innerText')
    pattern = r'\([^)]*\)'
    category = re.sub(pattern, '', category).strip()
    category = re.sub(r'/', '', category)
    return category

def move_product_save(page_num):
    global category_text, category_size, total_page, category_fail_cnt
    length = len(driver.find_elements(By.XPATH, '//*[@id="searchList"]/li'))
    product_a_lst = driver.find_elements(By.XPATH, '//*[@id="searchList"]/li/div[@class="li_inner"]/div[@class="article_info"]/p[@class="list_info"]/a')
    product_href_lst = list(map(lambda x: x.get_attribute('href'), product_a_lst))
    for k in tqdm(range(len(product_href_lst)), desc=f"{current_page} / {total_page} page crawling", position=1):
        product_href = product_href_lst[k]
        driver.get(product_href)
        src = driver.find_element(By.XPATH, '//*[@id="bigimg"]').get_attribute('src')
        response = requests.head(src)
        file_size = int(response.headers.get('Content-Length', 0))
        category_size += file_size * 1e-9

        img_lst = driver.find_elements(By.XPATH, '//div[@class="detail_product_info_item"]/descendant::img')#//*[@id="detail_view"]/div[1]
        for l, img in enumerate(img_lst):
            src = img.get_attribute('src')
            if not src:
                continue
            try:    
                response = requests.head(src)
                file_size = int(response.headers.get('Content-Length', 0))
                category_size += file_size * 1e-9
            except Exception as e:
                category_fail_cnt += 1

def move_page_product_crawling():
    global current_page, category_text
    page_len = len(driver.find_elements(By.XPATH, '//*[@id="goods_list"]/div[2]/div[1]/div/div/a[not(@aria-label)]'))
    for j in tqdm(range(page_len), desc=f"{category_text} crawling", position=0):
        current_page += 1
        page_a = driver.find_elements(By.XPATH, '//*[@id="goods_list"]/div[2]/div[1]/div/div/a[not(@aria-label)]')[j]
        page_href = page_a.get_attribute('href')
        page_a.send_keys(Keys.ENTER)
        move_product_save(j+1)
        driver.get(page_href)

def category_product_crawling():
    global current_page, total_page
    total_page = int(driver.find_element(By.CLASS_NAME, "totalPagingNum").get_attribute('innerText'))
    current_page = 0
    while True:
        url = driver.current_url
        move_page_product_crawling()
        driver.get(url)
        if total_page == current_page:
            break
        driver.find_element(By.CLASS_NAME, 'fa.fa-angle-right.paging-btn.btn.next').send_keys(Keys.ENTER)
        
global driver
options = webdriver.ChromeOptions()

options.add_experimental_option('excludeSwitches', ['enable-logging'])

driver = webdriver.Chrome(
    'C:/chromedriver.exe', 
    options=options
)

musinsa_url = 'https://www.musinsa.com/app/'
driver.get(musinsa_url)

# category list
category_len = len(driver.find_elements(By.XPATH, '//*[@id="accordion2"]/nav/div[2 <= position() and position() <= 6]/div[2]/ul/li/a'))
a_lst = driver.find_elements(By.XPATH, '//*[@id="accordion2"]/nav/div[2 <= position() and position() <= 6]/div[2]/ul/li/a')
category_text_lst = list(map(get_category_name, a_lst))
href_lst = list(map(lambda x: x.get_attribute('href'), a_lst))

global category_size, categories_dict, category_fail_cnt
categories_dict = {}

for i, href in enumerate(href_lst): # category_len
    category_size = 0
    category_fail_cnt = 0
    driver.get(href)
    global category_text
    category_text = category_text_lst[i]
    print(f"start crawling {category_text}")
    category_product_crawling()
    print(f"{category_text} size: {round(category_size, 8)} GB | download failed count: {category_fail_cnt}")
    categories_dict[category_text] = category_size

categories_name = []
categories_size = []
for category in categories_dict:
    categories_name += [category]
    categories_size += [categories_dict[category]]
categories_name += ['total']
categories_size += [sum(categories_size)]
categories_size_df = pd.DataFrame({'name': categories_name, 'size': categories_size})
categories_size_df.to_csv('./categories_size.csv', encoding='cp949')