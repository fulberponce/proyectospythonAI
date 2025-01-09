import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Inicializar Mediapipe para detección de la mano
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Obtener las dimensiones de la pantalla
screen_width, screen_height = pyautogui.size()

# Activar desactivando el Fail-Safe (opcional)
pyautogui.FAILSAFE = False

while True:
    # Leer un frame desde la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video")
        break

    # Convertir el frame a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el frame con Mediapipe para detectar las manos
    results = hands.process(rgb_frame)

    # Si se detecta una mano
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Dibujar los puntos de la mano en el frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener la posición del dedo índice (punto 8) y del pulgar (punto 4)
            index_finger = landmarks.landmark[8]
            thumb_finger = landmarks.landmark[4]

            h, w, _ = frame.shape
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)
            thumb_x = int(thumb_finger.x * w)
            thumb_y = int(thumb_finger.y * h)

            # Mapear las coordenadas del dedo a las coordenadas de la pantalla
            screen_x = np.interp(x, [0, w], [0, screen_width])
            screen_y = np.interp(y, [0, h], [0, screen_height])

            # Mover el cursor del mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Calcular la distancia entre el índice y el pulgar para detectar un "clic"
            distance = np.sqrt((x - thumb_x) ** 2 + (y - thumb_y) ** 2)

            # Si los dedos están suficientemente cerca (toco el pulgar y el índice), simular clic
            if distance < 50:  # Ajusta este valor según tus necesidades
                pyautogui.click()

    # Mostrar el frame con los puntos de la mano detectados
    cv2.imshow("Control de Mouse con Mano", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
