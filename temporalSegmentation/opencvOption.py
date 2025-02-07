import cv2
import numpy as np

# Captura de video
cap = cv2.VideoCapture("video_letra.mp4")

# Leer el primer frame
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (5,5), 0)

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (5,5), 0)

    # Calcular la diferencia absoluta entre frames
    diff = cv2.absdiff(frame1_gray, frame2_gray)

    # Aplicar un umbral para resaltar el movimiento
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Mostrar la imagen con movimiento segmentado
    cv2.imshow("Segmentación de Movimiento", thresh)

    # Actualizar el frame anterior
    frame1_gray = frame2_gray.copy()

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
