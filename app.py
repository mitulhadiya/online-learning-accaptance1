from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained pipeline
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form values
        age = int(request.form["age"])
        gender = request.form["gender"]
        device_type = request.form["device_type"]
        login_frequency_weekly = int(request.form["login_frequency_weekly"])
        avg_session_duration_min = float(request.form["avg_session_duration_min"])
        video_watch_time_min = float(request.form["video_watch_time_min"])
        avg_quiz_score = float(request.form["avg_quiz_score"])
        attendance_rate = float(request.form["attendance_rate"])
        final_grade = float(request.form["final_grade"])

        # Create DataFrame with EXACT column names
        data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "device_type": [device_type],
            "login_frequency_weekly": [login_frequency_weekly],
            "avg_session_duration_min": [avg_session_duration_min],
            "video_watch_time_min": [video_watch_time_min],
            "avg_quiz_score": [avg_quiz_score],
            "attendance_rate": [attendance_rate],
            "final_grade": [final_grade]
        })

        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "⚠️ High Risk of Student Dropout"
            color = "red"
        else:
            result = "✅ Student is Likely to Continue"
            color = "green"

        return render_template(
            "index.html",
            prediction_text=result,
            prediction_color=color,
            values=request.form
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            prediction_color="red",
            values=request.form
        )


if __name__ == "__main__":
    app.run(debug=True)