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
    img = tf.image.resize(imagen, [28, 28])
    img = tf.image.rgb_to_grayscale(img)
    img = tf.squeeze(img)
    img = tf.cast(img, tf.float32) / 255.0
    
    # Hacer la predicción
    prediccion = modelo.predict(np.expand_dims(img, axis=0))
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

        # Verificar y ajustar canales de color
        if img.shape[-1] == 4:  # Si tiene 4 canales (RGBA), convertir a 3 canales (RGB)
            img = img[..., :3]
        elif img.shape[-1] != 3:  # Si no tiene 3 canales (RGB)
            st.warning("La imagen no tiene 3 canales RGB. Asegúrate de dibujar en RGB.")

        # Redimensionar y convertir a escala de grises
        img = tf.image.resize(img, [28, 28]) # redimensionamos a 28x28 px, que es como hemos entrenado el modelo
        img = tf.image.rgb_to_grayscale(img) # se convierte a escala de grises si no lo está
        img = tf.squeeze(img)  # Asegurar que tenga solo un canal de color

        # Normalizar la imagen
        img = tf.cast(img, tf.float32) / 255.0

        # Hacer la predicción
        prediccion = modelo.predict(np.expand_dims(img, axis=0)) # agrega una dimensión adicional al principio de la matriz de la imagen, convirtiéndola de (28, 28) a (1, 28, 28).
        prediccion_numero = np.argmax(prediccion) # determina cual es la clasificacion predicha con mayor probabilidad. Devuelve el índice del valor máximo

        # Mostrar la predicción
        st.write(f"El modelo predice que el número dibujado es: {prediccion_numero}")
    else:
        st.write("Por favor, dibuja un número antes de presionar el botón 'Predecir'.")