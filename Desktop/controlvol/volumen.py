import cv2
import seguimientoManos as sm
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Configuración de la cámara
anchoCam, altoCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, anchoCam)
cap.set(4, altoCam)

# Inicialización del detector de manos y control de volumen
detector = sm.detectormnaos(maxManos=1, Confdeteccion=0.7)
dispositivos = AudioUtilities.GetSpeakers()
interfaz = dispositivos.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volumen = cast(interfaz, POINTER(IAudioEndpointVolume))
Rangovol = volumen.GetVolumeRange()
print("Rango de volumen:", Rangovol)

volMin = Rangovol[0]  # Volumen mínimo
volMax = Rangovol[1]  # Volumen máximo

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video")
        break

    frame = detector.encontrarmanos(frame)
    lista, bbox = detector.encontrarposicion(frame, dibujar=False)

    if len(lista) != 0:
        # Coordenadas del pulgar y del índice
        x1, y1 = lista[4][1], lista[4][2]
        x2, y2 = lista[8][1], lista[8][2]

        # Dibujo de un círculo en los puntos clave
        cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

        # Línea entre los dedos
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Distancia entre el pulgar y el índice
        distancia = np.hypot(x2 - x1, y2 - y1)
        print("Distancia:", distancia)

        # Conversión de la distancia a un rango de volumen
        vol = np.interp(distancia, [50, 200], [volMin, volMax])
        volumen.SetMasterVolumeLevel(vol, None)

        # Barra de volumen en pantalla
        volBar = np.interp(distancia, [50, 200], [400, 150])
        cv2.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(frame, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)

        # Porcentaje de volumen
        volPorcentaje = np.interp(distancia, [50, 200], [0, 100])
        cv2.putText(frame, f'{int(volPorcentaje)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Mostrar el video
    cv2.imshow("Control de Volumen", frame)

    # Salir con la tecla Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
