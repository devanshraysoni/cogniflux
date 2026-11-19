from flask import Flask,request,jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("stress_model.h5")

@app.route("/predict",methods=["POST"])
def predict():

data = request.json

features = np.array([[
data["latency"],
data["mouse_speed"],
data["click_rate"],
data["pause_duration"],
data["typing_speed"]
]])

prediction = model.predict(features)

return jsonify({"stress":float(prediction[0][0])*100})

if __name__ == "__main__":
app.run()