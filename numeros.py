import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Función para cargar el modelo
def cargar_modelo():
    try:
        modelo = tf.keras.models.load_model('models/pred_numeros.h5')
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

# Función para procesar la imagen y hacer la predicción
def procesar_prediccion(modelo, imagen):
    # Asegurarse de que la imagen sea en escala de grises y redimensionarla
    print(f"Dimensiones de la imagen original: {imagen.shape}")
    img = tf.image.resize(imagen, [28, 28])
    print(f"Dimensiones de la imagen redimensionada: {img.shape}")
    img = tf.image.rgb_to_grayscale(img)
    print(f"Dimensiones de la imagen en escala de grises: {img.shape}")
    img = tf.squeeze(img)
    print(f"Dimensiones de la imagen después de squeeze: {img.shape}")
    img = tf.cast(img, tf.float32) / 255.0
    
    # Hacer la predicción
    print(f"Dimensiones de la imagen antes de la predicción: {img.shape}")
    prediccion = modelo.predict(np.expand_dims(img, axis=0))
    print(f"Predicción obtenida: {prediccion}")
    prediccion_numero = np.argmax(prediccion)
    
    return prediccion_numero

# Cargar el modelo una vez al inicio de la aplicación
modelo = cargar_modelo()

# -----------------------------------------
# Configurar la interfaz de Streamlit
st.title("Reconocimiento de Dígitos Dibujados a Mano ✏️")
st.markdown("Dibuja un número en el lienzo de abajo y presiona el botón 'Predecir' para ver la predicción.")

# Tamaño del lienzo
CANVAS_SIZE = 192

# Crear un lienzo interactivo en Streamlit
canvas_result = st_canvas(
    fill_color="#000000",    # Color para llenar dentro del rectángulo
    stroke_width=10,         # Ancho del lápiz
    stroke_color="#FFFFFF",  # Color del lápiz (blanco)
    background_color="#000000",  # Color de fondo (negro)
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas",
)

# Botón de predicción
if st.button("Predecir"):
    if canvas_result.image_data is not None: # Verifica que hay algo dibujado
        # Obtener la imagen dibujada
        img = canvas_result.image_data.astype(np.uint8)

        print(f"Dimensiones de la imagen dibujada: {img.shape}")

        # Redimensionar y convertir a escala de grises
        img = tf.image.resize(img, [28, 28]) # redimensionamos a 28x28 px, que es como hemos entrenado el modelo
        print(f"Dimensiones de la imagen redimensionada: {img.shape}")
        img = tf.image.rgb_to_grayscale(img) # se convierte a escala de grises si no lo está
        print(f"Dimensiones de la imagen en escala de grises: {img.shape}")
        img = tf.squeeze(img)  # Asegurar que tenga solo un canal de color
        print(f"Dimensiones de la imagen después de squeeze: {img.shape}")

        # Normalizar la imagen
        img = tf.cast(img, tf.float32) / 255.0
        print(f"Dimensiones de la imagen después de normalización: {img.shape}")

        # Hacer la predicción
        prediccion_numero = procesar_prediccion(modelo, img)
        print(f"Número predicho: {prediccion_numero}")

        # Mostrar la predicción
        st.write(f"El modelo predice que el número dibujado es: {prediccion_numero}")
    else:
        st.write("Por favor, dibuja un número antes de presionar el botón 'Predecir'.")