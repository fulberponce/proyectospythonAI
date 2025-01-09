import cv2
import mediapipe as mp

# Inicializar Mediapipe para detección de rostros
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Iniciar la captura de video (por defecto usa la cámara 0)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abre correctamente
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara")
    exit()

print("Cámara abierta correctamente, comenzando a capturar...")

# Definir los índices de los puntos faciales que queremos conectar con líneas
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION  # Lista de conexiones entre puntos faciales

while True:
    # Leer un frame desde la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara")
        break

    # Obtener el tamaño del frame
    height, width, _ = frame.shape

    # Convertir el frame a RGB (Mediapipe usa imágenes en formato RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el frame con Mediapipe
    results = face_mesh.process(rgb_frame)

    # Dibujar los puntos faciales si se detectan
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar las líneas entre los puntos faciales
            for connection in FACE_CONNECTIONS:
                start = connection[0]
                end = connection[1]
                x1, y1 = int(face_landmarks.landmark[start].x * width), int(face_landmarks.landmark[start].y * height)
                x2, y2 = int(face_landmarks.landmark[end].x * width), int(face_landmarks.landmark[end].y * height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Dibujar los puntos faciales
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Mostrar el frame con los puntos y líneas faciales
    cv2.imshow("Detección de Rostros en Tiempo Real", frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
