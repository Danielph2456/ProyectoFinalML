from flask import Blueprint, request, render_template, jsonify
from app.preprocessing import preprocess
from app.model import predict

app_routes = Blueprint('app_routes', __name__)

@app_routes.route('/')
def index():
    return render_template('index.html')  # Renderiza la página HTML principal

@app_routes.route('/predict', methods=['POST'])
def predict_route():
    try:

        # Captura los datos enviados desde el formulario
        data = request.json        
        
        # Preprocesa los datos
        processed_data = preprocess(data)        

        # Realiza la predicción
        prediction = predict(processed_data)        
        
        # Devuelve la predicción como respuesta JSON
        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})
