import cv2
import numpy as np

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Cargar el clasificador Haar para detectar rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Crear un sustractor de fondo para la detección de movimiento
fgbg = cv2.createBackgroundSubtractorMOG2()

# Para almacenar las posiciones anteriores del rostro detectado
last_position = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Crear una máscara para la detección de movimiento
    fgmask = fgbg.apply(frame)

    # Encontrar los contornos en la máscara de movimiento
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Si detectamos rostros, dibujamos los rectángulos
    for (x, y, w, h) in faces:
        # Definir la región donde está el rostro
        face_region = frame[y:y+h, x:x+w]
        
        # Verificar si hay movimiento en la región del rostro
        movement_detected = False
        
        if last_position is not None:
            # Verificar si hay un desplazamiento significativo del rostro
            prev_x, prev_y, prev_w, prev_h = last_position
            if abs(x - prev_x) > 10 or abs(y - prev_y) > 10:  # Movimiento detectado
                movement_detected = True

        # Si no hay movimiento, dibujamos en verde; si hay movimiento, dibujamos en rojo
        color = (0, 255, 0) if not movement_detected else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Actualizar la posición del rostro
        last_position = (x, y, w, h)

    # Mostrar el video con los rectángulos dibujados
    cv2.imshow('Detección de Movimiento y Rostros', frame)

    # Salir si presionas 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
