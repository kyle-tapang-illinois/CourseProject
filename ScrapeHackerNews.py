# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:33:31 2022

@author: WilliamKiger
"""

import requests
import urllib.request
from urllib.error import HTTPError
from socket import timeout
from bs4 import BeautifulSoup
import pprint


COUNT = 1

def increment(): 
    global COUNT
    COUNT = COUNT + 1

def getPageText(url):

    print(url)
    increment()
    print(COUNT)
    
    if url.startswith(('http://', 'https://')):
        try:     
            html = urllib.request.urlopen(url, timeout=(3)).read()
        except HTTPError as e:
            print (e)
            return ''
        except timeout:
            print('timeout error occured')
            return ''
        
        soup = BeautifulSoup(html, features="lxml")

        # deactivate all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # extract the text

        # get the text ...then clean it up
        text = soup.get_text()

        #remove leading and trailing space https://stackoverflow.com/questions/1936466/how-to-scrape-only-visible-webpage-text-with-beautifulsoup
        lines = (line.strip() for line in text.splitlines())
        # break up headlines
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # eliminate blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # print(text.encode('utf-8'))
        return text.encode('utf-8')
          
    else:
        print("invalid url found is " + url)
        return ''      
    
def rankByVotes(data_list):
  return sorted(data_list, key= lambda k:k['votes'], reverse=True)

def extract_metadata(links, subtext, scraping_depth, search_external_bool): 
  data = []
  columns = len(links[0])
  
  for i in range(scraping_depth):
      for j in range(columns):
          # print("columns == " + str(columns))
          title = links[i][j].getText() 
          # print("title: " + title)
          href = links[i][j].a.get('href', None)
          
          #controls searching external to hacker news
          if search_external_bool == True :
              page_contents = getPageText(href)
          else:
              page_contents = '' #if not, contents will be empty
              
          vote = subtext[i][j].select('.score') #this is now subtext[num of pages][0-29 number of articles per page] 
          # print("vote: " + str(vote))
          
          if len(vote): #votes should be a list...means above line iw wrong
              points = int(vote[0].getText().replace(' points', ''))
              data.append({'title': title, 'link': href, 'votes': points, 'contents': page_contents})
        
  return rankByVotes(data)
 
def ScrapePages(url_list):
    
    mega_links = []
    mega_subtext = []
    
    for i in range(len(url_list)):
        #get request
        get_page = requests.get(url_list[i])
        soup = BeautifulSoup(get_page.text, 'html.parser')
        links = soup.select('.titleline') #grabbing <span class='titleline"> == $0 
        # print(links)
        subtext = soup.select('.subtext') #contains score under titleline
        # print(subtext)
        mega_links.append(links)
        mega_subtext.append(subtext)
        
    return mega_links, mega_subtext
        

def PagesToScrape(search_depth):
    # print("search depth is: " + str(search_depth))
    urls = []
    urls.append('https://news.ycombinator.com/news')
    
    if search_depth >= 2: 
        search_depth = search_depth - 1
        num = 2
        
        for i in range(search_depth): 
            page = 'https://news.ycombinator.com/news?p={}'.format(str(num))
            # print(page)
            urls.append(page)
            num = num + 1
            
    return urls


def main():
    
    #controls how many pages you want to scrape from Hacker News
    scraping_depth = 1
    
    #loading a list of urls for the depth of pages looked at in Hacker News
    urls = PagesToScrape(scraping_depth)
    print(urls)
    
    #now we are getting the Hacker news articles(not searching the actual articles yet): links and subtext
    links, subtext = ScrapePages(urls)
    
    #******this boolean will control going to the linked pages *******
    search_external_bool = False
    # search_external_bool = True
    
    #score_sorted is a dictionary with link, title, votes, contents. 
    # if search_external_bool is true, this program will grab the text from that page too and insert it in the contents of the score sorted dictionary
    score_sorted = extract_metadata(links, subtext, scraping_depth, search_external_bool)
    
    #this output will be massive if you are scraping pages external to Hacker News too 
    #which is controled by teh search_external_bool                                  
    pprint.pprint(score_sorted)
    

if __name__ == '__main__':
    main()
    
    
