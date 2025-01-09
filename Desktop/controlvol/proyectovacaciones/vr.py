import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math

# Generar puntos de una espiral en 4D
def generate_4d_spiral(num_points=500, a=1, b=1):
    points_4d = []
    for t in range(num_points):
        # Ecuación paramétrica para la espiral en 4D
        x = a * math.cos(t * 0.1)
        y = b * math.sin(t * 0.1)
        z = a * math.cos(t * 0.1) * math.sin(t * 0.1)
        w = b * math.sin(t * 0.1) * math.cos(t * 0.1)
        points_4d.append([x, y, z, w])
    return points_4d

# Proyección 4D a 3D
def project_4d_to_3d(vertex_4d):
    x, y, z, w = vertex_4d
    # Proyección simple de 4D a 3D, ignorando la dimensión w
    # Añadimos un pequeño ajuste si w es -1 para evitar la división por cero
    if w == -1:
        w = -0.999999
    scale_factor = 1 / (1 + w)  # Escala para dar efecto de acercamiento/lejanía
    x3d = x * scale_factor
    y3d = y * scale_factor
    z3d = z * scale_factor
    return (x3d, y3d, z3d)

# Dibujar la espiral en 3D proyectada
def draw_spiral():
    points_4d = generate_4d_spiral()
    points_3d = [project_4d_to_3d(v) for v in points_4d]

    glBegin(GL_LINE_STRIP)  # Usamos LINE_STRIP para dibujar la espiral
    for point in points_3d:
        glVertex3fv(point)
    glEnd()

# Inicialización de pygame y OpenGL
def init():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

# Función principal
def main():
    init()
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)  # Rota la espiral
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_spiral()
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
