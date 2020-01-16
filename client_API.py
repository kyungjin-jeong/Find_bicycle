from flask import Flask, request, jsonify, render_template, redirect
import traceback
import pandas as pd
import joblib
import sys

# Your API definition
app = Flask(__name__)
app.config.from_object('config')

@app.route('/web', methods=['GET', 'POST'])
def web():
    message = None
    
    if request.method == 'POST':
        speed = int(request.form['Bike_Speed'])
        cost = float(request.form['Cost_of_Bike'])
        season = str(request.form['Season'])
        time = str(request.form['Time'])
        city = str(request.form['City'])
        premise = str(request.form['Premise_Type'])
        bike = str(request.form['Bike_Type'])
        #color = str(request.form['Bike_Colour'])
        
        if speed != '' and cost != '' and season != '' and time != '' and city != '' and premise != '' and bike != '':
                
            df = pd.DataFrame({'Bike_Speed': [speed],
                               'Cost_of_Bike': [cost],
                               'Season': [season],
                               'Time': [time],
                               'City': [city],
                               'Premise_Type': [premise],
                               'Bike_Type': [bike],
                               #'Bike_Colour': [color]
                           })
            
            query = pd.get_dummies(df)
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(model.predict(query))
            
            if str(prediction[0] == 1.0): #Recovered
                message = 'Your bike is likely to be !' + str(prediction)
            elif str(prediction[0] == 0.0):
                message = 'Your bike is likely to be !' + str(prediction)
                
        else:
            message = 'Please enter all features'
            
    return render_template('predictForm.html', message = message)

@app.route("/json", methods=['GET','POST']) #use decorator pattern for the route
def json():
    if model:
        try:
            json_ = request.get_json()
            print('json_ : ', json_, '\n\n')
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(model.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to team2 model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    model = joblib.load('/Users/kyungjin/find_bicycle/final_model.pkl') # Load "final_model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('/Users/kyungjin/find_bicycle/final_model_columns.pkl') # Load "final_model_columns.pkl"
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)
