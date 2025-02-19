from PIL import ImageFont
font_path = '../fonts/Roboto-Regular.ttf'  # Ruta a la fuente personalizada
font_size = 28  # Tamaño de la fuente
DATA_DIR = './data'  # Carpeta para almacenar las imágenes
DATA_VIDEO_DIR = "./video" # Carpeta para almacenar los videos
dataset_size = 2000  # Número de imágenes por clase
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Letras del alfabeto

def load_font():
    try:
        return ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Custom font not found. Using default font.")
        return None
    