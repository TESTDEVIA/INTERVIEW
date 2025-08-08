import numpy
from pandas import read_csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
import sys
import requests
import urllib.request
from sty import fg, bg, ef, rs
import platform
import datetime as dt
from datetime import datetime
from dateutil import tz
import time
import dateparser
import pytz
import csv
from os.path import isfile, join
import warnings
warnings.filterwarnings("ignore")

start = time.time()

cachefolder="datamysql/"

def reverse(s):
    return s[::-1]

def load_csv(folder,symbol):
    global cachefolder
    cache = cachefolder+reverse(symbol)+".sql"
    # print ("Processing ",symbol,"...",sep='',end='')
    with open(folder+'cache2.csv', 'w') as f, \
            open(cache, 'r') as r:
        for line in r:
            f.write(line)

# Configuration

def server_ok(url):
    lOK=True
    try:
        ftpstream = urllib.request.urlopen(url)
    except:
        lOK=False
    return lOK

os.system("")

def ansiColors():
    if (platform.system()=="Windows") and (platform.release()=="7"):
        return False
    return True

def red(message):
    if ansiColors():
        return fg.li_red+message+fg.rs
    return message

def green(message):
    if ansiColors():
        return fg.li_green+message+fg.rs
    return message

def pink(message):
    if ansiColors():
        return fg.li_magenta+message+fg.rs
    return message

def blue(message):
    if ansiColors():
        return fg.li_cyan+message+fg.rs
    return message

def underlined(message):
    if ansiColors():
        return ef.u+message+rs.u
    return message

def elapsedTime(start):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds

def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)

def download_csv(folder,symbol):
    CSV_URL = 'https://bot.smartgemlab.com/download_symbol_cache?simbolo='+symbol
    with open(folder+'cache2.csv', 'wb') as f, \
            requests.get(CSV_URL, stream=True) as r:
        for line in r.iter_lines():
            f.write(line+'\n'.encode())

def filter_csv(folder,ts):
    with open(folder+'cache2.csv','r') as fin, open (folder+'cache2_filtered.csv','w') as fout:
        writer = csv.writer(fout, delimiter=',')            
        for row in csv.reader(fin, delimiter=','):
            if float(row[0]) >= ts:
                 writer.writerow(row)    

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

if not server_ok("https://bot.smartgemlab.com/ok"):
    print ("Server "+red('OFF'))
    print ("Process "+red('terminated'))
    quit()

def read_queue(cache):
    simbolos=[]
    neuronas=[]
    aepochs=[]
    periodos=[]
    carpetas=[]
    capas=[]
    status=[]
    if os.path.exists(cache):
        with open(cache, 'r') as fd:
            reader = csv.reader(fd)
            for row in reader:
                simbolos.append(row[0])
                neuronas.append(row[1])
                aepochs.append(row[2])
                periodos.append(row[3])
                carpetas.append(row[4])
                capas.append(row[5])
                status.append(row[6])
    return simbolos, neuronas, aepochs, periodos, carpetas, capas, status

def parse_symbol(cache,loss):
    simbolos=[]
    neuronas=[]
    aepochs=[]
    periodos=[]
    carpetas=[]
    capas=[]
    status=[]
    test=["1y","6m","1m","15d","10d","7d","3d"]
    p="7y"
    for t in test:
        if t in loss:
            p=t
    ctest=["2c","3c","4c","5c","7c"]
    ct="1"
    for c in ctest:
        if c in loss:
            ct=c.replace("c","")

    simbolos.append(cache)
    neuronas.append("4")
    aepochs.append("290")
    periodos.append(p)
    carpetas.append(loss)
    capas.append(ct)
    status.append("1")

    return simbolos, neuronas, aepochs, periodos, carpetas, capas, status

def upload_file(sfile):
    error=0
    test_file = open(sfile, "rb")
    test_url = "https://bot.smartgemlab.com/upload"
    try:
       test_response = requests.post(test_url, files = {"file": test_file})
       test_response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
        error=1
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
        error=1
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
        error=1
    except requests.exceptions.RequestException as err:
        error=1
        print ("OOps: Something Else",err)
    if error==1:
       error_file = open("error.sh", "a")
       error_file.write("python3 upload.py "+sfile+"\n")
       error_file.close()    

def upload_net(sfile):
    error=0
    test_file = open(sfile, "rb")
    test_url = "https://bot.smartgemlab.com/upload_net"
    try:
       test_response = requests.post(test_url, files = {"file": test_file})
       test_response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
        error=1
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
        error=1
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
        error=1
    except requests.exceptions.RequestException as err:
        error=1
        print ("OOps: Something Else",err)
    if error==1:
       error_file = open("error.sh", "a")
       error_file.write("python3 uploadnet.py "+sfile+"\n")
       error_file.close()    

def getTerm(s):
    nInterval=24        # long term interval is 24h
    if ("m" in s):
        nInterval=1     # med term interval is 1h
    if ("d" in s):
        nInterval=0     # short term interval is default (15m or data resolution)       
    return nInterval

def get_period(f, interval):
    t=os.path.getmtime(f)
    period=time.time() - t 
    period=int(period/3600)
    strperiod = "{:0>1} hours ago".format(period)
    print ("The net "+green(f.replace("_mean_squared_error_adam.h5",""))+" was trained "+green(strperiod))
    if period<96:
        print (blue("The net is up to date"))
        print ("Process "+red('terminated'))
        quit()
   
    return strperiod  

period="7 days ago"
symbol="SANTOSBTC"
loss="mean_squared_error"
optimizer="adam"
neurons=4
epochs=100
folder=""
layers=1
cache='cache.ini'

if (len(sys.argv)>=2):
   cache=sys.argv[1]
if (len(sys.argv)>=3):
   network=sys.argv[2]

if "." in cache:
    simbolos, neuronas, aepochs, periodos, carpetas, capas, status = read_queue(cache)
else:
    simbolos, neuronas, aepochs, periodos, carpetas, capas, status = parse_symbol(cache,network)

seen = set()
for c in carpetas:
    if c in seen:
        continue
    if not os.path.exists(c):
        os.makedirs(c)
    seen.add(c)


n=0
found=0
for s in simbolos:
    if (status[n]=="1") and (found==0):
        symbol=s
        neurons=int(neuronas[n])
        epochs=int(aepochs[n])
        period=periodos[n]
        period=period.replace("y"," years ago")
        period=period.replace("m"," months ago")
        period=period.replace("d"," days ago")        
        folder=carpetas[n]
        layers=int(capas[n])
        found=1
        status[n]="0"
    n=n+1


if len(folder)>0 and not folder.endswith("/"):
    folder=folder+"/"

label=folder.replace("/","")

interval=getTerm(folder)

if isfile(folder+symbol+"_"+loss+"_"+optimizer+'.h5'):
    period=get_period(folder+symbol+"_"+loss+"_"+optimizer+'.h5',interval)

if found==0:
    print (red("No data")+" to process")
    print ("Process "+red('terminated'))
    quit()

load_csv(folder,symbol)
filter_csv(folder,date_to_milliseconds(period))


numpy.random.seed(7)

dataframe = read_csv(folder+'cache2_filtered.csv', usecols=[4], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = neurons-1

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = Sequential()

if layers==1:
    model.add(LSTM(neurons, input_shape=(look_back, 1)))
else:    
    for xlayer in range(layers-1):
        model.add(LSTM(neurons, input_shape=(look_back, 1), return_sequences=True))
    model.add(LSTM(neurons, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss=loss, optimizer=optimizer)
for ie in range(0,epochs):
    print ("Epoch: ",ie+1)
    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=0)
    if ((ie+1) % 100 == 0) and (ie < epochs) and (epochs>=300):
        print ("Saving...")
        model.save(folder+label+"e"+str(ie+1)+"_"+symbol+"_"+loss+"_"+optimizer+'.h5')
        upload_file(folder+label+"e"+str(ie+1)+"_"+symbol+"_"+loss+"_"+optimizer+'.h5')

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

hours, minutes, seconds = elapsedTime(start)

print ()
print("Process terminated in ", end='', sep='')
if hours>0:
    print("{:0>2} hours ".format(int(hours)), end='', sep='')
if minutes>0:
    print("{:0>2} min ".format(int(minutes)), end='', sep='')
if seconds>0:
    print("{:05.2f} secs ".format(seconds), end='', sep='')
print()

model.save(folder+symbol+"_"+loss+"_"+optimizer+'.h5')
model.save(folder+label+"_"+symbol+"_"+loss+"_"+optimizer+'.h5')

upload_file(folder+label+"_"+symbol+"_"+loss+"_"+optimizer+'.h5')
upload_net(folder+label+"_"+symbol+"_"+loss+"_"+optimizer+'.h5')
os.remove(folder+label+"_"+symbol+"_"+loss+"_"+optimizer+'.h5')
