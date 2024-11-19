import numpy as np
import pandas as pd
import pickle
from flask import  Flask,render_template,request
app = Flask(__name__, static_url_path='/static')
df=pd.read_csv("ADANIPORTS.csv")
df.head(1)
x=df.iloc[:,3:8].values
y=df.iloc[:,8].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
from sklearn.svm import SVR
model = SVR(kernel='poly',degree=2,C=2)
model.fit(x_train, y_train)
with open ('stock.pkl','wb') as file:
    pickle.dump(model,file)
y_pred = model.predict(x_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
MAE = mean_absolute_error(y_test,y_pred)
print(MAE)
MSE = mean_squared_error(y_test,y_pred)
print(MSE)
RMSE = np.sqrt(MSE)
print(RMSE)
with open ('stock.pkl','rb') as file:
    model1 = pickle.load(file)
model1.predict([[440,770,1050,770,959]])
@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
     Prev_Close=float(request.form['Prev_Close'])
     Open = float(request.form['Open'])
     High = float(request.form['High'])
     Low = float(request.form['Low'])
     Last = float(request.form['Last'])
     result= model.predict([[Prev_Close,Open,High,Low,Last]])[0]
     return render_template('index.html',result="{}".format(result))
if __name__=="__main__":
    app.run(debug=True)