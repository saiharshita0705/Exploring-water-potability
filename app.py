# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model from the pickle file
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():

    result = None
    
    # Data to perform normalization
    min_arr = [3.902475686, 118.9885791, 320.9426113, 3.239580331, 268.6469407, 201.6197368, 5.362370906, 26.50548404, 1.872572601]
    max_arr = [10.25281623, 274.8283693, 44376.18738, 10.99999516, 399.6172172, 652.5375916, 23.31769912, 107.3063431, 6.083772354]
    if request.method == 'POST':
        
        # Get user inputs from the form
        input_data = [
            float(request.form['input1']),
            float(request.form['input2']),
            float(request.form['input3']),
            float(request.form['input4']),
            float(request.form['input5']),
            float(request.form['input6']),
            float(request.form['input7']),
            float(request.form['input8']),
            float(request.form['input9']),
        ]
        
        # Normalize the data
        for i in range(0, len(input_data)):
            input_data[i] = (input_data[i] - min_arr[i]) / (max_arr[i] - min_arr[i])
        
        # Reshape the input data to match the model's expectations
        input_array = np.array(input_data).reshape(1, -1)

        # Make a prediction using the machine learning model
        prediction = model.predict(input_array)

        # Display the result as "Yes" or "No"
        result = "Yes" if prediction[0] == 1 else "No"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
