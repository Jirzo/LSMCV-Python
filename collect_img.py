import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

# Configuración
font_path = './fonts/Roboto-Regular.ttf'  # Ruta a la fuente personalizada
font_size = 28  # Tamaño de la fuente
DATA_DIR = './data'  # Carpeta para almacenar las imágenes
dataset_size = 833  # Número de imágenes por clase
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Letras del alfabeto

# Crear directorio base si no existe
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Inicializar cámara
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

window_name = 'Capture Images'  # Nombre de la ventana

# Función para cargar fuente personalizada
def load_font():
    try:
        return ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Fuente personalizada no encontrada. Usando fuente predeterminada.")
        return None

font = load_font()

while True:
    # Seleccionar clase
    print("\nSeleccione la letra para capturar (A-Z). Ingrese 'Q' para salir.")
    selected_letter = input("Letra: ").strip().upper()

    if selected_letter == "Q":
        break

    if selected_letter not in alphabet:
        print("Entrada inválida. Por favor, seleccione una letra válida (A-Z).")
        continue

    class_dir = os.path.join(DATA_DIR, selected_letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'\nPreparando captura para la letra "{selected_letter}"...')
    text = f'Capturing: {selected_letter} | Press "Q" to stop'

    # Mostrar texto en pantalla hasta que se presione 'Q'
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el cuadro. Verifica la cámara.")
            break

        # Convertir frame OpenCV a imagen PIL
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        if font:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        else:
            text_width, text_height = 200, 30  # Estimado para texto básico

        # Posición del texto
        text_x, text_y = 20, 20
        padding = 10

        # Dibujar rectángulo de fondo
        rect_x0 = text_x - padding
        rect_y0 = text_y - padding
        rect_x1 = text_x + text_width + padding
        rect_y1 = text_y + text_height + padding
        draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0, 0, 0, 150))

        # Dibujar texto
        text_color = (255, 255, 255)  # Blanco
        draw.text((text_x, text_y), text, font=font or ImageFont.load_default(), fill=text_color)

        # Convertir imagen PIL a OpenCV
        frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Mostrar frame con texto
        cv2.imshow(window_name, frame_with_text)

        # Cerrar con 'Q'
        if cv2.waitKey(25) == ord('q'):
            break

    # Capturar imágenes para la letra seleccionada
    print(f"Comenzando captura para '{selected_letter}'...")
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el cuadro durante la colección de datos.")
            break

        cv2.imshow(window_name, frame)
        cv2.waitKey(25)

        # Guardar imagen con el formato <letra>_<número>.jpg
        filename = f'{selected_letter}_{counter:03d}.jpg'  # Formato del nombre
        filepath = os.path.join(class_dir, filename)
        cv2.imwrite(filepath, frame)
        counter += 1

        print(f"Imagen guardada: {filename}")

    print(f"Captura para la letra '{selected_letter}' completada.")

cap.release()
cv2.destroyAllWindows()
