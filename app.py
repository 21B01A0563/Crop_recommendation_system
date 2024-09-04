from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    input_values = [
        float(request.form['nitrogen']),
        float(request.form['phosphorous']),
        float(request.form['potassium']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]

    # Make a prediction using the loaded model
    recommended_crop = model.predict([input_values])[0]

    return render_template('result.html', crop=recommended_crop)

if __name__ == '__main__':
     app.run(host='0.0.0.0', debug=True)
