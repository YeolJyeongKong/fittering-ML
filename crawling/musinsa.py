from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains

import re
import os
import urllib.request
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

os.chdir('./crawling/crawling_data2')

def get_category_name(category_a):
    category = category_a.get_attribute('innerText')
    pattern = r'\([^)]*\)'
    category = re.sub(pattern, '', category).strip()
    category = re.sub(r'/', '', category)
    return category

def move_product_save(page_num):
    global category_text
    length = len(driver.find_elements(By.XPATH, '//*[@id="searchList"]/li'))
    product_a_lst = driver.find_elements(By.XPATH, '//*[@id="searchList"]/li/div[@class="li_inner"]/div[@class="article_info"]/p[@class="list_info"]/a')
    product_href_lst = list(map(lambda x: x.get_attribute('href'), product_a_lst))
    for k, product_href in enumerate(product_href_lst):
        driver.get(product_href)
        time.sleep(3)
        file_dir = category_text + '/' + f"{page_num}_{k}" + '/'
        os.makedirs(file_dir, exist_ok=True)

        src = driver.find_element(By.XPATH, '//*[@id="bigimg"]').get_attribute('src')
        urllib.request.urlretrieve(src, file_dir + "bigimg.png")

        img_lst = driver.find_elements(By.XPATH, '//div[@class="detail_product_info_item"]/descendant::img')#//*[@id="detail_view"]/div[1]
        print(len(img_lst))
        for l, img in enumerate(img_lst):
            src = img.get_attribute('src')
            if not src:
                continue
            urllib.request.urlretrieve(src, file_dir + f"{l}.png")

def move_page_product_crawling():
    global current_page
    page_a_lst = driver.find_elements(By.XPATH, '//*[@id="goods_list"]/div[2]/div[1]/div/div/a[not(@aria-label)]')
    page_href_lst = list(map(lambda x: x.get_attribute('href'), page_a_lst))
    for j, page_href in enumerate(page_href_lst):
        current_page += 1
        driver.get(page_href)
        move_product_save(j+1)

def category_product_crawling():
    total_page = int(driver.find_element(By.CLASS_NAME, "totalPagingNum").get_attribute('innerText'))
    global current_page
    current_page = 0
    while True:
        url = driver.current_url
        move_page_product_crawling()
        driver.get(url)
        print(current_page, total_page)
        if total_page == current_page:
            break
        driver.find_element(By.CLASS_NAME, 'fa.fa-angle-right.paging-btn.btn.next').send_keys(Keys.ENTER)
        
global driver
options = webdriver.ChromeOptions()

options.add_experimental_option('excludeSwitches', ['enable-logging'])

driver = webdriver.Chrome(
    'C:/chromedriver.exe', 
    options=options,
)

musinsa_url = 'https://www.musinsa.com/app/'
driver.get(musinsa_url)

# category list
category_len = len(driver.find_elements(By.XPATH, '//*[@id="accordion2"]/nav/div[2 <= position() and position() <= 6]/div[2]/ul/li/a'))
a_lst = driver.find_elements(By.XPATH, '//*[@id="accordion2"]/nav/div[2 <= position() and position() <= 6]/div[2]/ul/li/a')
category_text_lst = list(map(get_category_name, a_lst))
href_lst = list(map(lambda x: x.get_attribute('href'), a_lst))

for i, href in enumerate(href_lst): # category_len
    driver.get(href)
    global category_text
    category_text = category_text_lst[i]
    os.makedirs(category_text, exist_ok=True)
    print(category_text)
    category_product_crawling()
