from flask import Flask, render_template, request, redirect, url_for
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import gdown

app = Flask(__name__)

# Configuración para la carga de archivos
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Descargar el archivo CSV de Google Drive
file_id = '1AVLqLvgjue64xuT6okTdN0XnM2Gceccx'  # Reemplaza esto con tu FILE_ID
gdown_url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(gdown_url, 'data.csv', quiet=False)

# Cargar el DataFrame (con puntos clave)
keyfacial_df = pd.read_csv('data.csv')

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

        # Procesar la imagen
        process_image(filepath)
        return render_template('index.html', image_file='output.png')

def process_image(filepath):
    # Cargar la imagen original
    img = Image.open(filepath).convert('L')  # Convertir a escala de grises
    original_size = img.size  # Obtener las dimensiones originales de la imagen (ancho, alto)
    img_arr = np.array(img)  # Convertir la imagen a un array de NumPy

    # Crear una figura para graficar con los números de los ejes visibles
    fig, ax = plt.subplots(figsize=(6, 6))  # Tamaño ajustado de la imagen
    ax.imshow(img_arr, cmap='gray')
    
    # Seleccionar un índice aleatorio para los puntos faciales
    k = random.randint(0, len(keyfacial_df) - 1)  # Seleccionamos una fila aleatoria del DataFrame
    
    # Asumimos que las coordenadas de los puntos faciales están en una escala de 96x96 píxeles
    original_scale = 96  # Escala original del dataset de puntos faciales

    # Definir la región donde está el rostro (centrar la escala en una porción de la imagen)
    face_width = int(original_size[0] * 0.5)  # Ajustar el ancho de la cara como un 60% del ancho de la imagen
    face_height = int(original_size[1] * 0.5)  # Ajustar el alto de la cara como un 60% del alto de la imagen
    offset_x = (original_size[0] - face_width) // 2  # Calcular el desplazamiento en x para centrar la cara
    offset_y = (original_size[1] - face_height) // 2  # Calcular el desplazamiento en y para centrar la cara

    # Dibujar los puntos faciales en la imagen
    for j in range(1, 31, 2):  # Asumimos que hay 30 puntos (15 pares de x, y)
        x_original = keyfacial_df.loc[k][j-1]  # Coordenada X
        y_original = keyfacial_df.loc[k][j]    # Coordenada Y

        # Escalar los puntos faciales para que coincidan con las dimensiones de la región facial de la imagen
        x_scaled = (x_original / original_scale) * face_width + offset_x
        y_scaled = (y_original / original_scale) * face_height + offset_y

        # Graficar los puntos faciales en rojo ('rx') sobre la imagen
        ax.plot(x_scaled, y_scaled, 'rx', markersize=5)
    
    # Mostrar los ejes con números (coordenadas X, Y)
    ax.set_xlim(0, original_size[0])
    ax.set_ylim(original_size[1], 0)  # El eje Y está invertido en las imágenes

    # Guardar la imagen con los puntos faciales
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'output.png'), bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '_main_':
    app.run(debug=True)