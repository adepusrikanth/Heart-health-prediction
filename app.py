import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = list()
    print(request.form.keys())
    for key in request.form.keys():
        if key == "chest_pain_type" or key == "Rest_ECG" or key == "st_slope" or key == "thalassemia":
            val_string = request.form[key].split(",")
            for val in val_string:
                int_features.append(float(val))
        else:
            int_features.append(float(request.form[key]))
    #int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
	
    result=''
	
    if output == 0:
        result='is healthy '
    else:
        result='is not healthy ,should take care and consult doctor'
	
    return render_template('index.html', prediction_text='The Heart  {}'.format(result))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)