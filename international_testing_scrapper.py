#!/usr/bin/env python
# coding: utf-8


from tqdm import tqdm
import os
import pickle
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager  
import time
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def save_state(filename):
    with open('melbrune_files.pickle', 'wb') as state_file:
        pickle.dump(filename, state_file)


download_dir = "/Users/Shared/Files From d.localized/Maestria/tesis/MELU_IMAGES"

chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument("--headless")
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    # "plugins.always_open_pdf_externally": True  # If needed
})

driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)



# file_list = os.listdir(os.path.join('..', 'MELU_Annotation_Labels', 'MELU_Labels'))
# file_dict = {k:False for k in file_list}
with open('melbrune_files.pickle', 'rb') as handle:
    file_list = pickle.load(handle)

BASE_URL = 'https://online.herbarium.unimelb.edu.au/collectionobject/'

xpath_download_button = '/html/body/div[1]/div[2]/div/div/div[2]/div/div[1]/div[2]/a[2]'

def download_images(driver:webdriver,reporsitory_url:str,image_name:str,xpath_locator:str,waiting_times:list):
    driver.get(f'{reporsitory_url}/{image_name}')
    driver.implicitly_wait(waiting_times[0])
    download_button = driver.find_element_by_xpath(xpath_locator) 
    driver.execute_script("arguments[0].click();", download_button)
    driver.implicitly_wait(waiting_times[1])

for file,read in tqdm(file_list.items(), total=len(file_list)):
    images_id = file.split('_')[0]
    if images_id == 'MELUD15848b':
        print('here')
        continue
        #download_images(driver,BASE_URL,'MELUD121701c',xpath_download_button,[3,3])
    elif (len(images_id) > 4) & (read == False):   
        try:
            download_images(driver,BASE_URL,images_id,xpath_download_button,[3,3])
        except NoSuchElementException: 
            full_of_options = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div').text
            # print(full_list_of_options)
            full_list_of_options = full_of_options.split('\n')
            similar_list = [similar(x,images_id) for x in full_list_of_options]
            alternative_name = full_list_of_options[similar_list.index(max(similar_list))]
            if max(similar_list) > 0.88:
                download_images(driver,BASE_URL,alternative_name,xpath_download_button,[5,3])
            else :
                driver.refresh()        
        new_status = not read
        file_list[file] = new_status
        save_state(file_list)
    # time.sleep(2)
driver.quit()


