import numpy as np

class Transformacoes:
    @staticmethod
    def get_matriz_translacao(dx, dy):
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    @staticmethod
    def get_matriz_escalonamento(sx, sy):
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    @staticmethod
    def get_matriz_rotacao(angulo_rad):
        c = np.cos(angulo_rad)
        s = np.sin(angulo_rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def aplicar_transformacao_generica(self, obj, matriz_transformacao):
        coords_transformadas = []
        for x, y in obj.coords:
            ponto_original = np.array([x, y, 1])
            ponto_novo = np.dot(matriz_transformacao, ponto_original)
            coords_transformadas.append((ponto_novo[0], ponto_novo[1]))
        obj.coords = coords_transformadas

    def get_centro_objeto(self, obj):
        if not obj.coords:
            return (0, 0)
        x_coords = [p[0] for p in obj.coords]
        y_coords = [p[1] for p in obj.coords]
        cx = sum(x_coords) / len(x_coords)
        cy = sum(y_coords) / len(y_coords)
        return (cx, cy)

    def aplicar_escalonamento_natural(self, obj, sx, sy):
        cx, cy = self.get_centro_objeto(obj)
        matriz_t1 = self.get_matriz_translacao(-cx, -cy)
        matriz_s = self.get_matriz_escalonamento(sx, sy)
        matriz_t2 = self.get_matriz_translacao(cx, cy)
        matriz_final = np.dot(matriz_t2, np.dot(matriz_s, matriz_t1))
        self.aplicar_transformacao_generica(obj, matriz_final)

    def aplicar_rotacao_centro_objeto(self, obj, angulo_rad):
        cx, cy = self.get_centro_objeto(obj)
        matriz_t1 = self.get_matriz_translacao(-cx, -cy)
        matriz_r = self.get_matriz_rotacao(angulo_rad)
        matriz_t2 = self.get_matriz_translacao(cx, cy)
        matriz_final = np.dot(matriz_t2, np.dot(matriz_r, matriz_t1))
        self.aplicar_transformacao_generica(obj, matriz_final)

    def aplicar_rotacao_ponto_arbitrario(self, obj, angulo_rad, px, py):
        matriz_t1 = self.get_matriz_translacao(-px, -py)
        matriz_r = self.get_matriz_rotacao(angulo_rad)
        matriz_t2 = self.get_matriz_translacao(px, py)
        matriz_final = np.dot(matriz_t2, np.dot(matriz_r, matriz_t1))
        self.aplicar_transformacao_generica(obj, matriz_final)