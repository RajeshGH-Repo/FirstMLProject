from flask import  Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    model = pickle.load(open("SSmodel.pkl", "rb"))

    cgpa = float(request.form.get("cgpa"))
    iq = int(request.form.get("iq"))
    profile_score = int(request.form.get("profile_score"))

    data = np.array([cgpa,iq,profile_score]).reshape(1,3)
    result = model.predict(data)

    if int(result) == 1:
        result="Student will be placed"
    else:
        result="Student will NOT be placed"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
