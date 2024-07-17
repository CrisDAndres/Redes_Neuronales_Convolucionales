# Reconocimiento de dígitos manuscritos con Redes Neuronales Convolucionales
 
Este proyecto implementa redes neuronales convolucionales (CNN) utilizando TensorFlow y Keras para el reconocimiento de dígitos escritos a mano en el conjunto de datos MNIST. Se incluyen técnicas de aumento de datos (*Data Augmentation*) para mejorar el rendimiento del modelo y una interfaz de usuario simple desarrollada con Streamlit que permite dibujar un dígito en un lienzo y obtener una predicción del modelo.

## Estructura del proyecto 📂

El proyecto consiste en los siguientes archivos:

- ``notebook/``: Carpeta que contiene el notebook de Jupyter con el código utilizado para realizar el entrenamiento de la red neuronal, con explicaciones detalladas de cada paso.Folder available on the Google Drive link [Data](https://drive.google.com/drive/folders/1YNj80AnFaNC3GuXIMYGxBIITjxB3YKO6?usp=drive_link), containing the data files in csv format.

- ``models/``: Carpeta que contiene el modelo de la red neuronal entrenada en formato ``.keras``.

- ``numeros.py``: Script de Python para la aplicación de Streamlit.

## Tecnologías utilizadas

- ![TensorFlow](https://img.shields.io/badge/TensorFlow-14354C?style=for-the-badge&logo=tensorflow&logoColor=white): Biblioteca principal para la creación y entrenamiento de redes neuronales.
- ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white): API de alto nivel utilizada como interfaz para TensorFlow en este proyecto.
- ![Streamlit](https://img.shields.io/badge/Streamlit-2CA5E0?style=for-the-badge&logo=streamlit&logoColor=white): Framework utilizado para desarrollar la interfaz web interactiva.
- ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white): Utilizado para el desarrollo y documentación del proceso de entrenamiento del modelo.
- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white): Lenguaje de programación en el cual está implementado todo el proyecto.
  
## Descripción del proyecto 📝

El dataset MNIST contiene un conjunto de 60k imágenes de entrenamiento y 10k imágenes de prueba de dígitos manuscritos (de 0 a 9) en escala de grises, cada una de 28x28 píxeles. Es ampliamente utilizado en el entrenamiento de sistemas de procesamiento de imágenes y modelos de aprendizaje automático.

Este proyecto se centra en la creación de una Red Neuronal Convolucional (CNN) utilizando **TensorFlow**, una de las bibliotecas más utilizadas en el campo del aprendizaje automático y la inteligencia artificial. Las **CNN** son muy efectivas para tareas de clasificación de imágenes y reconocimiento de objetos, ya que pueden identificar y aprender características espaciales de las imágenes de manera eficiente. 

Para hacer el modelo más accesible e interactivo, se ha desarrollado una interfaz web sencilla usando Streamlit. Esto permite a los usuarios dibujar un dígito en un lienzo y recibir la predicción del modelo en tiempo real.

## Instrucciones de ejecución 💻
Para ejecutar este proyecto en tu máquina local, sigue los siguientes pasos:

0. Asegúrate de tener instalado Python XX o superior.
1. Clona este repositorio en tu máquina local.
2. Descarga las carpetas ``models`` y ``notebook``, así como el scrip de Python.
3. Instala las dependencias necesarias ejecutando ``pip install -r requirements.txt``.
4. Ejecuta el archivo ``numeros.py`` y asegúrate de que has descargado las carpetas en el mismo entorno. A continuación, abre la terminal y ejecuta el siguiente comando: ``streamlit run numeros.py``. Esto abrirá el navegador web ``http://localhost:8501/`` que te llevará a la aplicación.
5. Dibuja✏️ un dígito del 0 al 9 en el lienzo y presiona el botón de **PREDECIR** para ver la predicción. 

## Contacto 📧
Si tienes alguna pregunta o sugerencia sobre este proyecto, no dudes en ponerte en contacto conmigo. Puedes ponerte en contacto conmigo a través de mis redes sociales.
