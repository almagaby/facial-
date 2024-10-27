import os
import threading
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import gdown
import cv2  # OpenCV para detección de rostros
import dlib  # Dlib para puntos faciales

app = Flask(_name_)

# Configuración para la carga de archivos
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Descargar archivo CSV (solo si no existe)
csv_file = 'data.csv'
if not os.path.exists(csv_file):
    file_id = '1vJPJZU88lo6nFSQC9i9e4y-eUujJfXia'    
    gdown_url = f'https://drive.google.com/file/d/1vJPJZU88lo6nFSQC9i9e4y-eUujJfXia/view?usp={file_id}'
    
    gdown.download(gdown_url, csv_file, quiet=False)

# Cargar puntos faciales del DataFrame
keyfacial_df = pd.read_csv(csv_file)

# Ruta al predictor de puntos faciales (asegúrate de que el archivo exista)
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Precargar el detector y el predictor una vez al iniciar
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Procesar la imagen en un hilo separado
        threading.Thread(target=process_image, args=(filepath,)).start()
        return render_template('index.html', image_file='output.png')

def process_image(filepath):
    """Procesa la imagen cargada y guarda los puntos faciales."""
    # Reducir resolución para acelerar el procesamiento (máx 800x800)
    img = Image.open(filepath).convert('L')
    img.thumbnail((800, 800))  # Redimensionar sin perder proporción
    img_arr = np.array(img)

    # Detectar rostros
    faces = face_detector(img_arr)
    if len(faces) == 0:
        print("No se detectó ningún rostro.")
        return  # Detener si no hay rostros

    # Usar el primer rostro detectado
    face = faces[0]
    landmarks = landmark_predictor(img_arr, face)

    # Crear la figura y graficar los puntos faciales
    plt.figure(figsize=(img.width / 100, img.height / 100))
    plt.imshow(img_arr, cmap='gray')
    plt.axis('off')

    # Definir los puntos clave que queremos mostrar (cejas, ojos, nariz y boca)
    # Cejas: 2 puntos en cada una (17, 21 para ceja izquierda; 22, 26 para ceja derecha)
    key_points = [17, 21, 22, 26]  
    # Ojos: 3 puntos en cada uno (36, 39, 37 para ojo izquierdo; 42, 45, 43 para ojo derecho)
    key_points += [36, 39, 37, 42, 45, 43]  
    # Nariz: 1 punto central (33)
    key_points += [30]  
    # Boca: 4 puntos clave (48, 54 en las comisuras, 51 en el labio superior, 57 en el labio inferior)
    key_points += [48, 54, 51, 57]

    # Graficar todos los puntos faciales en rojo
    for n in key_points:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        plt.plot(x, y, 'rx', markersize=5)  # "x" roja en todos los puntos

    # Guardar la imagen procesada
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if _name_ == '_main_':
    app.run(debug=True)
