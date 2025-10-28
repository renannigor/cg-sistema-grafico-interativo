
import numpy as np
from typing import List, Tuple

class Ponto3D:
    def __init__(self, x, y, z):
        self.coords = np.array([x, y, z, 1.0])

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        return self.coords[2]

    def aplicar_transformacao(self, matriz: np.ndarray):
        self.coords = matriz @ self.coords


class Transformacoes3D:
    @staticmethod
    def get_matriz_translacao(dx, dy, dz):
        return np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])

    @staticmethod
    def get_matriz_escalonamento(sx, sy, sz):
        return np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])

    @staticmethod
    def get_matriz_rotacao_x(angulo_rad):
        c = np.cos(angulo_rad)
        s = np.sin(angulo_rad)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def get_matriz_rotacao_y(angulo_rad):
        c = np.cos(angulo_rad)
        s = np.sin(angulo_rad)
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def get_matriz_rotacao_z(angulo_rad):
        c = np.cos(angulo_rad)
        s = np.sin(angulo_rad)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    @staticmethod
    def _normalizar(vetor):
        norma = np.linalg.norm(vetor)
        return vetor / norma if norma != 0 else vetor

    @classmethod
    def get_matriz_rotacao_eixo_arbitrario(cls, p1, p2, angulo_rad):
        eixo = cls._normalizar(np.array(p2) - np.array(p1))
        a, b, c = eixo
        d = np.sqrt(b**2 + c**2)

        T = cls.get_matriz_translacao(-p1[0], -p1[1], -p1[2])
        T_inv = cls.get_matriz_translacao(p1[0], p1[1], p1[2])

        Rx = np.identity(4)
        if d != 0:
            Rx = np.array(
                [
                    [1, 0, 0, 0],
                    [0, c / d, -b / d, 0],
                    [0, b / d, c / d, 0],
                    [0, 0, 0, 1],
                ]
            )
        Rx_inv = Rx.T

        Ry = np.array([[d, 0, -a, 0], [0, 1, 0, 0], [a, 0, d, 0], [0, 0, 0, 1]])
        Ry_inv = Ry.T

        Rz = cls.get_matriz_rotacao_z(angulo_rad)

        return T_inv @ Rx_inv @ Ry_inv @ Rz @ Ry @ Rx @ T