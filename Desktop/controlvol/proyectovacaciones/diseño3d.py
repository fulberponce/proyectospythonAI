import cv2
import mediapipe as mp
import pygame
from pygame import display
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

# Inicializar Mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Inicializar Pygame y la ventana de OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# Configuración inicial de la proyección de OpenGL
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Función para dibujar un cubo 3D
def draw_cube():
    glBegin(GL_QUADS)
    # Cara frontal
    glColor3f(1, 0, 0)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    
    # Cara trasera
    glColor3f(0, 1, 0)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, -1, -1)
    
    # Cara superior
    glColor3f(0, 0, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)
    
    # Cara inferior
    glColor3f(1, 1, 0)
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)
    
    # Cara derecha
    glColor3f(1, 0, 1)
    glVertex3f(1, -1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, -1, 1)
    
    # Cara izquierda
    glColor3f(0, 1, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)
    
    glEnd()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer el frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video")
        break

    # Convertir la imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el frame con Mediapipe
    results = hands.process(rgb_frame)

    # Si se detectan manos
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Dibujar los puntos de la mano
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Tomar las coordenadas de las manos (por ejemplo, índice y pulgar)
            index_finger = landmarks.landmark[8]
            thumb_finger = landmarks.landmark[4]
            
            # Obtener las coordenadas relativas a la imagen
            h, w, _ = frame.shape
            x_index = int(index_finger.x * w)
            y_index = int(index_finger.y * h)
            x_thumb = int(thumb_finger.x * w)
            y_thumb = int(thumb_finger.y * h)

            # Calcular el ángulo de rotación en función de la posición de la mano
            rotation_x = (x_index - x_thumb) / 100.0  # Ajusta según el rango de movimiento
            rotation_y = (y_index - y_thumb) / 100.0  # Ajusta según el rango de movimiento
            
            # Aplicar la rotación al cubo
            glRotatef(rotation_x, 1, 0, 0)  # Rotación alrededor del eje X
            glRotatef(rotation_y, 0, 1, 0)  # Rotación alrededor del eje Y
    
    # Dibujar el cubo
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_cube()
    
    # Actualizar la pantalla de OpenGL
    pygame.display.flip()
    pygame.time.wait(10)

    # Mostrar la imagen con los puntos de la mano detectados
    cv2.imshow("Control de 3D con Mano", frame)

    # Salir si presionas la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
