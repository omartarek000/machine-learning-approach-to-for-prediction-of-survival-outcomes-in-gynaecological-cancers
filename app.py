from flask import Flask, render_template, request, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import joblib
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///samples.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


model = joblib.load('models/gradient_boosting_model_4may.pkl')



class Sample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sample_id = db.Column(db.String(50), nullable=False)
    sample_date = db.Column(db.DateTime, default=datetime.utcnow)
    features = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<Sample {self.sample_id}>'

with app.app_context():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        csv_file = request.files.get('csv_file')
        
        try:
            df = pd.read_csv(csv_file)
            if df.shape[1] != 190:
                return render_template('error.html', error="CSV file must have 190 columns.")
            
            # Transform data with scaler
            predictions_raw = model.predict(df)
            probabilities = model.predict_proba(df)
            class_names = model.classes_

            # Handle multiple predictions
            predictions = []
            for i in range(len(df)):
                max_prob_idx = probabilities[i].argmax()
                highest_prob = probabilities[i][max_prob_idx]
                predicted_class = predictions_raw[i]
                predictions.append({
                    'row_number': i + 1,
                    'predicted_class': predicted_class,
                    'highest_prob': f"{highest_prob * 100:.2f}%"
                })

            # If single row, use the original template format
            if len(predictions) == 1:
                return render_template('predict.html', 
                                    single_row=True,
                                    predicted_class=predictions[0]['predicted_class'],
                                    highest_prob=predictions[0]['highest_prob'])
            
            # If multiple rows, use the table format
            return render_template('predict.html', 
                                single_row=False,
                                predictions=predictions)

        except Exception as e:
            return render_template('error.html', error=f"Error processing CSV file: {str(e)}")
    
    return render_template('index.html')

@app.route('/download_features')
def download_features():
    return send_file('models/features_template.csv', as_attachment=True, download_name='features_template.csv')

@app.after_request
def add_header(response):
    if app.debug:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    else:
        response.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return response

if __name__ == '__main__':
    app.run()