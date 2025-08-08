import numpy
from pandas import read_csv
import matplotlib.pyplot as plt
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import load_model 
import time
import sys
import requests
import urllib.request
from sty import fg, bg, ef, rs
import platform
import datetime as dt
from datetime import datetime
import time
import dateparser
import pytz
from dateutil import tz
import csv
from os.path import isfile, join
import warnings
warnings.filterwarnings("ignore")

OUTPUT="/reporte/cache/output/"
cachefolder="datamysql/"

start = time.time()

from_zone = tz.gettz('UTC')
to_zone = tz.gettz('America/Caracas')

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

# Configuration

def server_ok(url):
    lOK=True
    try:
        ftpstream = urllib.request.urlopen(url)
    except:
        lOK=False
    return lOK

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

def reverse(s):
    return s[::-1]

def load_csv(symbol):
    global cachefolder
    cache = cachefolder+reverse(symbol)+".sql"
    # print ("Processing ",symbol,"...",sep='',end='')
    with open('predict.csv', 'w') as f, \
            open(cache, 'r') as r:
        for line in r:
            f.write(line)

def download_csv(folder, symbol):
    global cachefolder
    CSV_URL = 'https://bot.smartgemlab.com/download_symbol_cache?simbolo='+symbol
    print ("Descargando simbolo ",symbol,"...",sep="")
    with open('predict.csv', 'wb') as f, \
            requests.get(CSV_URL, stream=True) as r:
        for line in r.iter_lines():
            f.write(line+'\n'.encode())

def timestamptodate (when):
    ts=int(when/1000.0)
    utc = dt.datetime.fromtimestamp(ts)
    utc = utc.replace(tzinfo=from_zone)
    when = utc.astimezone(to_zone)
    ret = when.strftime("%d/%m/%Y %I:%M:%S%p")
    
    return ret

def timestamptodate00(ts):
    return datetime.utcfromtimestamp(ts/1000).strftime('%d-%m-%Y')+" "+datetime.utcfromtimestamp(ts/1000).strftime("%I:%M:%S%p")

def filter_csv(folder, ts, interval):
    minimo=date_to_milliseconds("1 Jan,2038")
    maximo=0
    diff=0
    inter=0
    with open('predict.csv','r') as fin, open ('predict_filtered.csv','w') as fout:
        writer = csv.writer(fout, delimiter=',')            
        for row in csv.reader(fin, delimiter=','):
            if len(row)>3 and float(row[0]) >= ts:
                x=float(row[0])
                if inter>=interval:
                    writer.writerow(row) 
                    inter=0   
                if x>maximo:
                    diff=x-maximo
                    maximo=x
                if x<minimo:
                    minimo=x
                inter=inter+0.25
    if False:
        print ("Ventana de Tiempo:", sep='')
        print (timestamptodate(minimo), sep='')
        print (timestamptodate(maximo), sep='')
        print ("Interval: ",interval, sep='')
        print (diff/3600000, sep='')
    return minimo, maximo, diff

def load_btc(folder):
    btc = []
    with open(folder+'BTCBUSD.csv','r') as fin:
        for row in csv.reader(fin, delimiter=','):
            btc.append(float(row[2])) 
    return btc


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def get_period(f, interval):
    t=os.path.getmtime(f)
    period=time.time() - t 
    period=int(period/3600)
    strperiod = "{:0>1} hours ago".format(period)
    print ("The net "+green(f.replace("_mean_squared_error_adam.h5",""))+" was trained "+green(strperiod))
    if period<3:
        strperiod = "3 hours ago"
    if interval==24:
        strperiod = "1 months ago"        
    if interval==1:
        strperiod = "15 days ago"        
    return strperiod   

def getTerm(s):
    nInterval=24        # long term interval is 24h
    if ("m" in s):
        nInterval=1     # med term interval is 1h
    if ("d" in s):
        nInterval=0     # short term interval is default (15m or data resolution)       
    return nInterval

    
# Configuration
REMOTE_CACHE=0
if not server_ok("https://bot.smartgemlab.com/ok"):
    print ("Servidor remoto "+red('deshabilitado'))
    print ("Proceso "+red('terminado'))
    quit()

response = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
data = response.json()
BTC=data["bpi"]["USD"]["rate"]
BTC=BTC.replace(",","")
BTC=float(BTC)

symbol="SANTOSBTC"
period=""
folder = ""

if (len(sys.argv)>=2):
   symbol=sys.argv[1]
if (len(sys.argv)>=3):
   folder=sys.argv[2]
if (len(sys.argv)>=4):
   period=sys.argv[3]

if len(folder)>0 and not folder.endswith("/"):
    folder=folder+"/"


interval=getTerm(folder)

if period=="":
   period=get_period(folder+symbol+'.h5',interval)

if period=="":
    if isfile(folder+folder.replace("/","")+"_"+symbol+'.h5'):
        period=get_period(folder+folder.replace("/","")+"_"+symbol+'.h5',interval)
    else:
        period=get_period(folder+symbol+'.h5',interval)

print ("Prediction time: "+pink(period))

load_csv(symbol)

minimo, maximo, diff = filter_csv(folder,date_to_milliseconds(period),interval)

if interval>0:
    diff=interval*3600000

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('predict_filtered.csv', usecols=[4], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

def predict(close_data, num_prediction, model):
    prediction_list = close_data[-1:]
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = numpy.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(df, num_prediction):
    last_date = df['open_time'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

# load model
if isfile(folder+folder.replace("/","")+"_"+symbol+'.h5'):
    model = load_model(folder+folder.replace("/","")+"_"+symbol+'.h5')
else:
    model = load_model(folder+symbol+'.h5')

# summarize model.
model.summary()

# print ("trainX")
# print ("------")
# print (trainX)

# forecast_dates = predict_dates(dataframe, num_prediction)

# quit()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# default
num_prediction = 60      # 15 hours = 60 x 15min
if interval==24:
    num_prediction = 30  # 30 days  = 30 x 24h
    print (green("Long term prediction: 30 days"))
if interval==1:
    num_prediction = 24  # 24 hours = 24 x 1h
    print (pink("Medium term prediction: 24 h"))
if interval==0:
    print (blue("Short term prediction: 15h / 15 min"))

forecastPredict = predict(testX,num_prediction, model)
forecastPredict = forecastPredict.reshape(-1, 1)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
forecastPredict = scaler.inverse_transform(forecastPredict)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

hours, minutes, seconds = elapsedTime(start)

print ()
print("Process succesfully ended in ", end='', sep='')
if hours>0:
    print("{:0>2} hours ".format(int(hours)), end='', sep='')
if minutes>0:
    print("{:0>2} minutes ".format(int(minutes)), end='', sep='')
if seconds>0:
    print("{:05.2f} seconds".format(seconds), end='', sep='')
print()

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# shift forecast predictions for plotting
forecastPredictPlot = numpy.empty_like(dataset)
forecastPredictPlot[:, :] = numpy.nan

label=folder.replace("/","")

html_file=open(OUTPUT+symbol+"_"+label+'.html',mode="w",encoding="utf-8")
csv_file=open(folder+symbol+"_"+label+'.csv',mode="w",encoding="utf-8")
csv_acum=open(folder+symbol+"_"+label+'_acum.csv',mode="a",encoding="utf-8")

html_file.write("<style> td { color:white }</style>\n")
html_file.write("<table border='1'>\n")

lHayBTC=False
if os.path.exists(folder+"BTCBUSD.csv"):
    lHayBTC=True

if symbol.endswith("BTC") and lHayBTC:
    btc=load_btc(folder)
    # print (btc)
    html_file.write("<tr><td><b>Fecha/Hora</b></td><td><b>Prediccion BTC</b></td><td><b>Prediccion BTC/BUSD</b></td><td><b>Prediccion BUSD</b></td></tr>\n")
else:
    html_file.write("<tr><td><b>Fecha/Hora</b></td><td><b>Prediccion</b></td></tr>\n")

# add forecast predictions for plotting

# maximo = maximo - 14400000 # Zona Horaria Caracas
d = maximo
n=0
for f in forecastPredict:
    forecastPredictPlot = numpy.append(forecastPredictPlot, f) 
    d = d + diff
    strf = "{:.12f}".format(f[0])
    if symbol.endswith("BTC") and lHayBTC:
        html_file.write("<tr><td>"+timestamptodate(d)+"</td><td>"+strf+"</td><td>"+str(btc[n])+"</td><td>"+str(btc[n]*float(f[0]))+"</td></tr>\n")
        csv_file.write(str(d)+","+timestamptodate(d)+","+strf+"\n")
        csv_acum.write(str(d)+","+timestamptodate(d)+","+strf+"\n")
    else:
        html_file.write("<tr><td>"+timestamptodate(d)+"</td><td>"+strf+"</td></tr>\n")
        csv_file.write(str(d)+","+timestamptodate(d)+","+strf+"\n")
        csv_acum.write(str(d)+","+timestamptodate(d)+","+strf+"\n")
    n = n + 1

label=folder.replace("/","")

html_file.write("</table>\n")
html_file.write("<p>Bitcoin: "+str(BTC)+" "+timestamptodate(maximo)+"</p>\n")
html_file.close()
csv_file.close()
csv_acum.close()
upload_file(folder+symbol+"_"+label+'.csv')
upload_file(folder+symbol+"_"+label+'_acum.csv')
upload_file(OUTPUT+symbol+"_"+label+'.html')



# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset),label="Real")
plt.xlabel(symbol+ ", trainScore ="+str(trainScore)+", testScore ="+str(testScore)) # X axis data label
plt.plot(trainPredictPlot,label="Entrenamiento")
plt.plot(testPredictPlot,label="Prueba", linestyle='dashed')
plt.plot(forecastPredictPlot,label="Predicción", color="orange")
plt.title(symbol,loc="center")
plt.legend(loc='best')
plt.savefig(folder+"predict_"+symbol+"_"+label+'.png')

