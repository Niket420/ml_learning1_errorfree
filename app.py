
import pickle                                     ### anything u want to import u need to write it in requirements.txt file or download it in that file
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


## import ridge regressor model and standard scaler pickel file
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

## route for home page
@app.route('/')
def index():
    return render_template('index_html')         #render template will search for index.html file in the workspace 


@app.route('/predictdata',methods=['GET','POST'])          #predictdata is the new url for the our page...add predictdata after colon    #get means taking output and post means giving input
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))      # this is the way of code .....the variable written in bracket will match with the html code then redirect here
        RH = float(request.form.get('RH'))                      # order of the variable should be same as that of during training
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classses = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled =standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classses,Region]])
        result = ridge_model.predict(new_data_scaled)                   # it will return an array

        return render_template('home.html',result = result[0])            # the lhs result is the variable to access the home.html file
    else:                                                                  # render_template gives only one output thats why properly mentioned result[0]                             
        return render_template('home.html')             # if method is PUT then it will excute this









if __name__=="__main__":
    app.run(host="0.0.0.0")           # by default port is 5000 but if u want to change the port then write port =xgsgs in the app.run bracket
