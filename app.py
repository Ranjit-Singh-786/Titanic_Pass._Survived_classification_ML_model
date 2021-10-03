import joblib
from flask import Flask,render_template,request
from flask_cors import cross_origin
import sklearn
import pandas
import numpy
import jsonify
import pickle
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model=joblib.load('titanic_model_save_with_84%_accuracy')
@app.route('/')            #,methods=['GET']
@cross_origin()
def Home():
    return render_template('index.html')
# standard_to = StandardScaler()

@app.route("/predict",methods=["GET","POST"])           #,methods=['POST']
@cross_origin()
def predict():
    if request.method == 'POST':
        pclass=int(request.form['pclass'])
        age_of_passenger=float(request.form['age'])
        sibsp=int(request.form['sibsp'])
        parch=int(request.form['parch'])
        female=int(request.form['sex_female'])
        male=int(request.form['sex_male'])
        embark_C=int(request.form['embarked_c'])
        embark_Q=int(request.form['embarked_q'])
        embark_S=int(request.form['embarked_s'])

        town_c=int(request.form['embark_town_Cherbourg'])
        town_q=int(request.form['embark_town_Queenstown'])
        town_s=int(request.form['embark_town_Southampton'])
        prediction=model.predict([[pclass,age_of_passenger,sibsp,parch,female,male,embark_C,embark_Q,embark_S,town_c,town_q,town_s]])
        output=round(prediction[0],2)
        if output <1:
            return render_template('death.html',prediction_text=str(output))        
        else:
            return render_template('survived.html')

if __name__ =='__main__':
    app.run(debug=True)

















# without scaled data model
# independent-12
# dependent-1
# pclass has-3 category----1,2,3
# age columns is float and min 1 and max values is 54
# and other all columns are boolearn type
# sequnce of column
# ['pclass', 'age', 'sibsp', 'parch', 'survived', 'sex_female', 'sex_male',
#        'embarked_C', 'embarked_Q', 'embarked_S', 'embark_town_Cherbourg',
#        'embark_town_Queenstown', 'embark_town_Southampton']