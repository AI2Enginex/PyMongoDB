import pandas as pd
import time
import os
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By

'''Web Scraping using Selenium python we will use Google Chrome web browser for scraping the text from the website'''

# creating a class to scrape
# the hyperlinks from the
# main page
class WebScraping:

    def __init__(self,link=None):
        
        # declearing the variable for chrome web driver

        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager("131.0.6778.205").install()))
        self.driver.get(link)
    
    # function to release the 
    # chrome driver
    def releasedriver(self):

        self.driver.quit()
    
    # extracting all the links
    # from the web page
    def get_links(self,value2=None,value3=None):

        try:
            search = self.driver.find_elements(By.CLASS_NAME,value=value2)
            return [ele.text for data in search for ele in data.find_elements(By.TAG_NAME,value3) if len(ele.text) > 0]
        except:
            self.releasedriver()
    
    # searching and extracting names
    # within the hyperlinks
    def get_hyper_links(self,value2=None,value3=None,refrence=None):

        try:
            link_names = self.get_links(value2,value3)
            return [link_data.get_attribute(refrence) for text in link_names for link_data in 
                                                   self.driver.find_elements(By.LINK_TEXT, text)]
        except:
            self.releasedriver()

# class for creating a Dataframe
# to store the links
# these links will be used
# later to extract the text

class GetDataFrame(WebScraping):

    def __init__(self,link=None,val2=None,val3=None,ref=None):

        super().__init__(link)
        self.link_data = self.get_hyper_links(val2,val3,ref)
    
    # generating a dataframe with the links
    def getdataframe(self):

        try:
            return pd.DataFrame(self.link_data,index=range(len(self.link_data)),columns=['Links'])
        except Exception as e:
            return e
    
    # storing the dataframe into a csv file
    def createfile(self,filename=None):

        try:
            df = self.getdataframe()
            if os.path.isfile(filename):

                df.to_csv(filename,mode='a',index=False,header=False)
                print('append successful')
                
            else:
                df.to_csv(filename,index=False)
        except Exception as e:
            return e
        
# creating the class
# to read the data 
# in the form of dataframe
class ReadData:

    def __init__(self,file=None):
        
        # declearing a variable to read the dataframe
        self.df = pd.read_csv(file,encoding='utf-8')

        # dropping the duplicate values
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
    
    # creating a function to filter the dataframe
    # based on some condition
    def get_data(self,feature_name=None,str_lst=None,endswith_condition=None):

        try:
            filtered_df = self.df[self.df[feature_name].str.contains("|".join(str_lst))]
            if endswith_condition is True:
                return filtered_df[filtered_df[feature_name].str.endswith('.html')]
            else:
                return filtered_df
        except Exception as e:
            return e
        
    # function to read the data
    # from the feature
    def readdata(self,feature_name=None,str_lst=None,condition=None):

        try:
            filtered_df = self.get_data(feature_name,str_lst,condition)
            return filtered_df[feature_name].to_list()
        except Exception as e:
            return e

# creating a class
# to extract data
# from the stored links
class ExtractText:

    def __init__(self,link=None):
        
        # declearing the driver variable
        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager("131.0.6778.205").install()))
        self.driver.get(link)

    def releasedriver(self):

        self.driver.quit()
    
    # extracting the news title
    def news_title(self,title_val=None,tag_value=None):

        try:
            news_title = self.driver.find_elements(By.CLASS_NAME,value=title_val)
            return [news_.text for data in news_title for news_ in data.find_elements(By.TAG_NAME,tag_value)]
        except:
            self.releasedriver()
    
    # extracting news date and time
    # with the main content
    def get_text(self,value=None,value2=None,value3=None,value4=None):

        try:
            search = self.driver.find_elements(By.CLASS_NAME,value=value)
            return [ele.text for data in search for ele in data.find_elements(By.CLASS_NAME,value4) if len(ele.text) > 0], [datatext.text for data in search for data_ in 
                    data.find_elements(By.CLASS_NAME,value=value2) 
                    for datatext in data_.find_elements(By.TAG_NAME , value=value3)] 
        except:
            self.driver.quit()



# class for execueting and storing 
# the extraction of the hyper links
class GetLinks:

    def __init__(self,csv_file_name=None):

        self.r = ReadData(file=csv_file_name)
        
    def get_links(self,featurename=None,str_lst=None,endcondition=None):
        try:
            self.data = self.r.readdata(featurename,str_lst,endcondition)
            return self.data
        except Exception as e:
            return e



# creating a class to
# automatically store the 
# news,date,title as a
# document in MongoDB Database
class CreateDocuments:

    def __init__(self,file_name=None,feature=None,val_lst=None,end_con=None):
        
        # declearing a variable to read the links
        self.gl = GetLinks(csv_file_name=file_name)

        # storing the data into a list
        self.data = self.gl.get_links(featurename=feature,str_lst=val_lst,endcondition=end_con)

    def check_data_links(self):
        return self.data
    

    # function for creating a list of dictionaries
    # which will be inserted later as a document in the mongodb'c collection
    def get_documents(self,start=None,end=None,title_val=None,tag_val=None,val4=None,val=None,val2=None,val3=None):
        
        try:
            
            # declaring a list
            scraped_data = list()
            if end=='all':
                end = len(self.data)
            else:
                end = end
            for i in range(start,end+1):
                
                # class to extract text
                ex = ExtractText(link=self.data[i])
                title_list = ex.news_title(title_val,tag_val)
                date,text=ex.get_text(value4=val4,value=val,value2=val2,value3=val3)

                # formatting the title, date and time and the text
                # to be inserted in the mongodb's collection
                title_str = " ".join(title_list).strip() if title_list else ""
                dates = list(set(date)) if date else []
                date_str = " ".join(dates).strip() if dates else ""
                text_str = " ".join(text).strip() if text else "" 
                scraped_data.append({
                              "title": title_str,
                              "date and time": date_str,
                              "text": text_str
                            })
            return scraped_data
        except Exception as e:
            return e
        
if __name__ == '__main__':
    
    pass