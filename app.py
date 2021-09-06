from flask import Flask, request,render_template,jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import time

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',' Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)','Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis','Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis','Impetigo']


def MultinomialNBMethod(a):
    if not a :
        return {}
    df=pd.read_csv("CSV/Training.csv")
    cols= df.columns
    cols= cols[:-1]
    X= df[cols]
    y = df[["prognosis"]]
    s = time.time()
    mulNB = MultinomialNB()
    mulNB=mulNB.fit(X,np.ravel(y))
   
    return {'result':mulNB.predict(getSymtopms(a.split(','),cols))[0],'score':mulNB.score(X, np.ravel(y)),'time':time.time()-s}


def GaussianNBMethod(a):
    if not a :
        return {}
    df=pd.read_csv("CSV/Training.csv")
    cols= df.columns
    cols= cols[:-1]
    X= df[cols]
    y = df[["prognosis"]]
    s = time.time()
    gaussiannb = GaussianNB()
    gaussiannb=gaussiannb.fit(X,np.ravel(y))

    return {'result':gaussiannb.predict(getSymtopms(a.split(','),cols))[0],'score':gaussiannb.score(X, np.ravel(y)),'time':time.time()-s}

def KmeansMethod(a):
    if not a :
        return {}
    df=pd.read_csv("CSV/Training.csv")
    cols= df.columns
    cols= cols[:-1]
    X= df[cols]
    s = time.time()
    kmeans = KMeans(random_state=40,n_clusters=40,)
    kmeans=kmeans.fit(np.array(X))
   
    return {'result':disease[kmeans.predict(getSymtopms(a.split(','),cols))[0]],'score':kmeans.score(X),'time':time.time()-s}




def KNeighborsClassifierMethod(a):
    if not a :
        return {}
    df=pd.read_csv("CSV/Training.csv")
    cols= df.columns
    cols= cols[:-1]
    X= df[cols]
    y = df[["prognosis"]]
    s = time.time()
    kneighbor = KNeighborsClassifier()
    kneighbor.fit(X, np.ravel(y))
    

    return {'result':kneighbor.predict(getSymtopms(a.split(','),cols))[0],'score':kneighbor.score(X, np.ravel(y)),'time':time.time()-s}
     
def LogisticRegressionMethod(a):
    if not a :
        return {}
    df=pd.read_csv("CSV/Training.csv")
    cols= df.columns
    cols= cols[:-1]
    X= df[cols]
    y = df[["prognosis"]]
    s = time.time()
    logisticregression= LogisticRegression()
    logisticregression.fit(X, (y))
   
    return {'result':logisticregression.predict(getSymtopms(a.split(','),cols))[0],'score':logisticregression.score(X, np.ravel(y)),'time':time.time()-s}


def LogisticRegressionMethodv0(a):
    if not a :
        return {}
    df=pd.read_csv("CSV/Training.csv")
    cols= df.columns
    cols= cols[:-1]
    X= df[cols]
    y = df[["prognosis"]]
    logisticregression= LogisticRegression()
    logisticregression.fit(X, (y))
   
    return (logisticregression.predict(getSymtopms(a.split(','),cols))[0])


def getSymtopms(psymptoms,cols):
    l2=[]
    for x in range(0,132):
        l2.append(0)
    for k in range(0,132):
        for z in psymptoms:
            if(z==cols[k]):
                l2[k]=1
    return [l2]

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/api/getDiease/multinomialnb')
def multinomialnb():
    return jsonify(MultinomialNBMethod(request.args.get('Symptom')))

@app.route('/api/getDiease/kmeans')
def kmeans():
    return jsonify(KmeansMethod(request.args.get('Symptom')))

@app.route('/api/getDiease/kneighbors')
def kneighbors():
    return jsonify(KNeighborsClassifierMethod(request.args.get('Symptom')))

@app.route('/api/getDiease/logisticregression')
def logisticregression():
    return jsonify(LogisticRegressionMethod(request.args.get('Symptom')))

@app.route('/api/getDiease/gaussiannb')
def gaussiannb():
    return jsonify(GaussianNBMethod(request.args.get('Symptom')))

@app.route('/api/getDiease')
def getdieas():
    return jsonify(LogisticRegressionMethodv0(request.args.get('Symptom')))

if __name__ == '__main__':
    app.run(debug=True)