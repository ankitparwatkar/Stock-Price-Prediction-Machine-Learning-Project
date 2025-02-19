from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your model
model = pickle.load(open("stock.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify(prediction.tolist())

if __name__ == "__main__":
    app.run(debug=True)
