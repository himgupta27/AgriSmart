"""

import requests
from pprint import pprint

iot=requests.get('http://api.openweathermap.org/data/2.5/forecast?q=Rajpura&APPID=9e91ec239c87aebe08e82e1314463919&units=metric')
iodat=iot.json()

#print(iot)
#pprint(iodat)

weath0=iodat['list'][0]['weather'][0]['main']
weath1=iodat['list'][8]['weather'][0]['main']
weath2=iodat['list'][16]['weather'][0]['main']
weath3=iodat['list'][24]['weather'][0]['main']
weath4=iodat['list'][32]['weather'][0]['main']

print(weath1,weath2,weath3,weath4,weath0)

"""
import pandas as pd

fdata=pd.read_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/disease_feedback.csv", delimiter=",")
print(fdata)
lrow=fdata.tail(1)
print(lrow)
sno=int(lrow[fdata.columns[0]])
print(sno)
disease_details='ABC'
l=[sno+1,disease_details,'None']
print()
sno=sno+1

fdata.loc[sno]=[sno,'Y','None']
print(fdata)

#fdata.to_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/disease_feedback.csv")
