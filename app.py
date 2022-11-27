import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np

from sklearn.preprocessing import StandardScaler


app=Flask(__name__)

#final_model = pickle.load(open('final_model.pkl'))
final_model = pickle.load(open('final_model.pkl','rb'))
# with open('final_model.json', 'r') as f:
#     final_model = json.load(f)
pca_transform=pickle.load(open('pca.pkl','rb'))
#scalar = pickle.load(open('pca.pkl','rb'))
scalar = StandardScaler()
@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     #print(data)
#     #print(np.array(list(data.values())).reshape(1,-1))
#     pca_trans = pca_transform.transform([np.array[data]])
#     new_data=scalar.transform(np.array(list(pca_trans.values())))
#     output=final_model.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])
    
    #return data

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    
    inputs = scalar.fit_transform(np.array(list(data)).reshape(1, -1))
    
    pca_trans = pca_transform.transform(np.array(list(inputs)).reshape(1, -1))
    output = final_model.predict(pca_trans)
    output = int(output[0])
    return render_template("home.html",prediction_text="Your trip duration will be {} minutes".format((round(output,2))))
    #return data


if __name__=="__main__":
    app.run(debug=True)
   
     
