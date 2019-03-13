from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import requests
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pprint import pprint
import tensorflow as tf
import csv

from imageapp.models import Imageclass

def homepage(request):
    b1=Imageclass.objects.get(name=1)
    b2=Imageclass.objects.get(name=2)
    b3=Imageclass.objects.get(name=4)
    b4=Imageclass.objects.get(name='bg')
    b5=Imageclass.objects.get(name='bk')
    b8=Imageclass.objects.get(name='sky')
    return render(request,'home.html',{'obj1':b1, 'obj2':b2, 'obj3':b3, 'bg':b4, 'bk': b5, 'sky':b8})

def convert_image_to_array(image_dir,default_image_size=tuple((256, 256))):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def my_input(url):
    label_binarizer_classes_=['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
       'Potato___Early_blight', 'Potato___Late_blight',
       'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
       'Tomato_Late_blight', 'Tomato_Leaf_Mold',
       'Tomato_Septoria_leaf_spot',
       'Tomato_Spider_mites_Two_spotted_spider_mite',
       'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
       'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']    
    model = Sequential()
    inputShape = (256,256,3)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (3,256,256)
        chanDim = 1
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(15))
    model.add(Activation("softmax"))

    opt = Adam(lr=1e-3, decay=1e-3/25)
    model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

    model.load_weights("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/weights_test_cnn.hdfs")

    image=list()
    test=convert_image_to_array(url)
    image.append(test)
    np_image_test = np.array(image, dtype=np.float16)/225.0
    predictions_1 = model.predict(np_image_test)
     
    predictions=np.argmax(predictions_1,1)
    predictions[0]
    tf.keras.backend.clear_session()
    return label_binarizer_classes_[predictions[0]]


def diseasepage(request):
    myfile = request.FILES['input_img']
    mypic = request.POST.get("input_img")
    fs = FileSystemStorage()
    filename = fs.save(myfile.name, myfile)
    uploaded_file_url = fs.url(filename)
    path="C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/"+uploaded_file_url.replace('%20',' ')[6:]
    disease_details=my_input(path)
    fdata=pd.read_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/disease_feedback.csv", delimiter=",")
    lrow=fdata.tail(1)
    sno=int(lrow[fdata.columns[0]])
    sno=sno+1
    fdata.loc[sno]=[sno,disease_details,'None']
    fdata.to_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/disease_feedback.csv", index=False)

    treat='None'
    try:
        filename='C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/cure/'+disease_details+'.txt'
        f = open(filename, "r")
        treat=f.read()
    except:
        treat='Exception'

    if treat=='': treat='It is a healthy plant!'

    b7=Imageclass.objects.get(name='yg')

    return render(request, 'dpage.html',{'ourpic':uploaded_file_url, 'disease':disease_details, 'cure': treat, 'yg': b7, 'number':sno}) 

def feedback(request):
    fdata=pd.read_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/disease_feedback.csv", delimiter=",")
    sno=int(request.GET['ref'])
    val=request.GET['feed']
    l=fdata.loc[sno]
    l[2]=val
    fdata.loc[sno]=l
    fdata.to_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/disease_feedback.csv", index=False)
    return HttpResponse('<h1>Thank You for your feedback.</h1><h1>Your response has been noted and will be used to improve subsequent predictions.</h1>')

def predict_crop(l):
    data=pd.read_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/crop_prediction_datasets.csv", delimiter=",")
    row=data.shape[0]
    data.loc[row]=l
    data2=pd.get_dummies(data[['State_Name','pH','Moisture','Soil_temp', 'District_Name','Season','irrigation','Weather_1','Weather_2','Weather_3','Weather_4']])
    data2.to_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/crop_encoded.csv")
    
    X=pd.read_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/crop_encoded.csv")
    Y=pd.read_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/crop_prediction2.csv")
    #print(Y)
    del X['Unnamed: 0']
    del Y['0']
    
    X_data=X.values
    Y_data=Y.values
    
    row=X_data.shape[0]
    col=X_data.shape[1]


    x_train=X_data[0:64496,:]
    x_test=X_data[row-1:row,:]
    y_train=Y_data



    clf=DecisionTreeClassifier(criterion="entropy")
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    d={0: 'Arecanut', 1: 'Arhar/Tur', 2: 'Bajra', 3: 'Banana', 4: 'Barley', 5: 'Beans & Mutter(Vegetable)', 6: 'Bhindi', 7: 'Bottle Gourd', 8: 'Brinjal', 9: 'Cabbage', 10: 'Cashewnut', 11: 'Castor seed', 12: 'Citrus Fruit', 13: 'Coconut ', 14: 'Coriander', 15: 'Cotton(lint)', 16: 'Cowpea(Lobia)', 17: 'Cucumber', 18: 'Dry chillies', 19: 'Dry ginger', 20: 'Garlic', 21: 'Ginger', 22: 'Gram', 23: 'Grapes', 24: 'Groundnut', 25: 'Guar seed', 26: 'Horse-gram', 27: 'Jowar', 28: 'Jute', 29: 'Korra', 30: 'Lemon', 31: 'Linseed', 32: 'Maize', 33: 'Mango', 34: 'Masoor', 35: 'Mesta', 36: 'Moong(Green Gram)', 37: 'Moth', 38: 'Niger seed', 39: 'Oilseeds total', 40: 'Onion', 41: 'Orange', 42: 'Other  Rabi pulses', 43: 'Other Fresh Fruits', 44: 'Other Kharif pulses', 45: 'Other Vegetables', 46: 'Papaya', 47: 'Peas  (vegetable)', 48: 'Peas & beans (Pulses)', 49: 'Pome Fruit', 50: 'Pome Granet', 51: 'Potato', 52: 'Ragi', 53: 'Rapeseed &Mustard', 54: 'Rice', 55: 'Safflower', 56: 'Samai', 57: 'Sannhamp', 58: 'Sapota', 59: 'Sesamum', 60: 'Small millets', 61: 'Soyabean', 62: 'Sugarcane', 63: 'Sunflower', 64: 'Sweet potato', 65: 'Tapioca', 66: 'Tobacco', 67: 'Tomato', 68: 'Total foodgrain', 69: 'Turmeric', 70: 'Urad', 71: 'Varagu', 72: 'Water Melon', 73: 'Wheat', 74: 'other fibres', 75: 'other misc. pulses', 76: 'other oilseeds'}
    ans=[d[i] for i in y_pred]
    return(ans)

def soilanalysispage(request):
    
    if request.GET['manual']=='Yes':
        ph=float(request.GET['ph'])
        hum=float(request.GET['Humidity'])
        temp=float(request.GET['Temperature'])
        
    else:
        iot=requests.get('https://api.thingspeak.com/channels/714929/feeds.json?results=2')
        iodat=iot.json()['feeds'][-1]
        ph=iodat['field3']
        hum=iodat['field1']
        temp=iodat['field2']
    
    irr=request.GET['Irrigation'].lower()
    state=request.GET['State']
    district=request.GET['District']
    season=request.GET['Season']

    url = 'http://api.openweathermap.org/data/2.5/forecast?q={}&APPID=406e0137603a69be44bee1e86b44881d&units=metric'.format(district)
    #9e91ec239c87aebe08e82e1314463919
    #406e0137603a69be44bee1e86b44881d
    #00f0194010b2f7dcb2084122dc09eee1

    response=requests.get(url)
    data=response.json()

    weath1=data['list'][8]['weather'][0]['main']
    weath2=data['list'][16]['weather'][0]['main']
    weath3=data['list'][24]['weather'][0]['main']
    weath4=data['list'][32]['weather'][0]['main']

    
    b6=Imageclass.objects.get(name='sab')

    if weath1=='Mist': weath1='haze'
    elif weath1=='Clouds': weath1='cloudy'
    elif weath1=='Rain': weath1='rainy'
    else: weath1='sunny'
    

    if weath2=='Mist': weath2='haze'
    elif weath2=='Clouds': weath2='cloudy'
    elif weath2=='Rain': weath2='rainy'
    else: weath2='sunny'
    

    if weath3=='Mist': weath3='haze'
    elif weath3=='Clouds': weath3='cloudy'
    elif weath3=='Rain': weath3='rainy'
    else: weath3='sunny'
    

    if weath4=='Mist': weath4='haze'
    elif weath4=='Clouds': weath4='cloudy'
    elif weath4=='Rain': weath4='rainy'
    else: weath4='sunny'
    

    crop=predict_crop([0,state, district.upper(), season, irr,ph,hum,temp,weath1,weath2,weath3,weath4])[0]

    
    def demand():
        data=pd.read_csv("C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/demand.csv",delimiter=",")
        del data['irrigation']
        del data['ph']
        del data['Moisture']
        del data['temp']
        #data.head()

        crop_name=data['crop']
        total_crop=list(crop_name)
        #total_crop
        year=['yr_1','yr_2','yr_3','yr_4','yr_5']
        for yr in year:
            col="rate"+"_"+yr
            data[col]=np.random.randint(25,75,data.shape[0])

        data['avg']=(data['rate_yr_1']+data['rate_yr_2']+data['rate_yr_3']+data['rate_yr_4']+data['rate_yr_5'])/5
        data['dem']=(data['rate_yr_1']+1.2*data['rate_yr_2']+1.4*data['rate_yr_3']+1.6*data['rate_yr_4']+1.8*data['rate_yr_5']-5*data['avg'])/5
        data=data.sort_values('dem')
        #print(data.head)
        l=[]
        l.append(data.iloc[76]['crop'])
        l.append(data.iloc[75]['crop'])
        l.append(data.iloc[74]['crop'])

        #print(l)
        return l


    def final(l,crop):
        if(l[0]==crop):
            return crop
        if(l[1]==crop):
            return crop
        if(l[2]==crop):
            return crop
        return [crop,l[0],l[1],l[2]]

    def intercropping(l):
        Data = [[],[],[],[],[],[],[],[],[],[]]

        with open('C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/name_of_crop - name_of_crop.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader :
                if(int(row[1])!=10):
                    Data[int(row[1])].append(row[0])

        f=[]
        t=[]
        for crop in l:
            i=0
            for row in Data:
                if(crop in row ) :
                    break
                i=i+1
            t.append(i)
        #print(t)
        if t[0] == 10 :
            f.append(l[0])
        else :
            f.append(l[0])
            r=t[0]
            t.remove(t[0])
            i=0
            for ind in t :
                if ind!=r and ind!=10:
                    f.append(l[i+1])
                    break
                i=i+1
        stri='Primary Crop: '+f[0]
        if len(f)>1: stri=stri+' & Secondary Crop: '+f[1]
        return stri

    newans=intercropping(final(demand(),crop))
    
    betinfo=''

    try:
        filename='C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/Info/'+crop+'_'+state+'.txt'
        f = open(filename, "r")
        betinfo=f.read()
    except:
        pass


    info='Nothing'
    try:
        filename='C:/Users/Rajan Sethi/Desktop/Online Courses/mangobite/mangobite/Info/'+crop+'.txt'
        f = open(filename, "r")
        info=f.read()
    except:
        info='Exception'

    return render(request,'sapage.html',{'pH':ph, 'Humidity':hum, 'Temperature':temp, 'Irrigation':irr, 'Season':season, 'Crop':newans, 'State':state, 'District':district, 'Forecast':data['list'][8]['weather'][0]['description'], 'Information':betinfo, 'sab': b6, 'Infob':info})
    