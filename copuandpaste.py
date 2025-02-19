import shutil
import os

# Asegúrate de que el directorio de destino exista
destino_dir = "video/ÑrEnames/"
os.makedirs(destino_dir, exist_ok=True)  # Crea el directorio si no existe

# Extensión del video
video_extension = ".mp4"

for i in range(101):  # range() itera de 0 a 19, no hasta 20
    new_video_number = str(i).zfill(3)  # zfill(3) asegura 3 dígitos con ceros
    nombre_original = "video/Ñ/movimiento_4.mp4"  # Asegúrate de incluir la extensión .mp4
    nuevo_nombre = f"Ñ_00{i}.mp4"  # Incluye la extensión .mp4 en el nuevo nombre
    destination_video_path = os.path.join(destino_dir, f"Ñ_{new_video_number}{video_extension}")

    try:
        shutil.copy2(nombre_original, destination_video_path)  # Usa copy2 para copiar metadatos
        print(f"Archivo copiado y renombrado como: {nuevo_nombre}")
    except FileNotFoundError:
        print(f"Error: Archivo original no encontrado: {nombre_original}")
    except Exception as e:
        print(f"Error inesperado: {e}")