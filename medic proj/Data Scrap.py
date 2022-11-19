
import pickle
import re
from googlesearch import search
import warnings
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup
import time


small_alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
diseases=[]
for c in small_alpha:
    URL = 'https://www.nhp.gov.in/disease-a-z/'+c
    time.sleep(1)
    page = requests.get(URL,verify=False)

    soup = BeautifulSoup(page.content, 'html5lib')
    all_diseases = soup.find('div', class_='all-disease')

    for element in all_diseases.find_all('li'):
        diseases.append(element.get_text().strip())

with open('list_diseaseNames.pkl', 'rb') as handle:
    diseases2 = pickle.load(handle)



a=set(diseases)
b=set(diseases2)
c=list(a.union(b))
c.sort()


dis_symp={}
for dis in c:
  query = dis+' wikipedia'
  for sr in search(query,tld="co.in",stop=10,pause=0.5): 
    match=re.search(r'wikipedia',sr)
    filled = 0
    if match:
      wiki = requests.get(sr,verify=False)
      soup = BeautifulSoup(wiki.content, 'html5lib')
      info_table = soup.find("table", {"class":"infobox"})
      if info_table is not None:
        for row in info_table.find_all("tr"):
          data=row.find("th",{"scope":"row"})
          if data is not None:
            data=data.get_text()
            if data=="Symptoms":
              symptom=str(row.find("td"))
              symptom = symptom.replace('.','')
              symptom = symptom.replace(';',',')
              symptom=re.sub(r'<b.*?/b>:',',',symptom) 
              symptom=re.sub(r'<a.*?>','',symptom)
              symptom=re.sub(r'</a>','',symptom)
              symptom=re.sub(r'<[^<]+?>',', ',symptom) 
              symptom=re.sub(r'\[.*\]','',symptom)
              symptom=' '.join([x for x in symptom.split() if x != ','])
              dis_symp[dis]=symptom
              filled = 1
              break
    if filled==1:
      break
      

temp_list=[]
tmp_dict=dict()
for key,value in dis_symp.items():
  if value not in temp_list:
    tmp_dict[key]=value
    temp_list.append(value)
  else:
    print(key)

dis_symp = tmp_dict
print(len(dis_symp))
with open('final_dis_symp.pickle', 'wb') as handle:
   pickle.dump(dis_symp, handle, protocol=pickle.HIGHEST_PROTOCOL)
