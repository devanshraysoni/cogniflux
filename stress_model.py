import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("stress_dataset.csv")

X = data[["latency","mouse_speed","click_rate","pause_duration","typing_speed"]]
y = data["stress"]

model = RandomForestRegressor()

model.fit(X,y)

joblib.dump(model,"stress_model.pkl")