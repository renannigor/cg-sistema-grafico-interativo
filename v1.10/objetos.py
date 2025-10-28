# objetos.py
import numpy as np
from enum import Enum
from typing import List, Tuple
from geometria_3d import Ponto3D, Transformacoes3D


class TipoObjeto(Enum):
    PONTO = "ponto"
    RETA = "reta"
    POLIGONO = "poligono"
    CURVA = "curva"
    B_SPLINE = "b-spline"
    OBJETO_3D = "objeto_3d"
    SUPERFICIE_BEZIER = "superficie_bezier"
    SUPERFICIE_BSPLINE = "superficie_bspline"


class Objeto:
    def __init__(
        self,
        nome: str,
        tipo: str,
        coords: List[Tuple[float, float]],
        cor: str = "black",
        preenchido: bool = False,
    ):
        self.nome = nome
        self.tipo = tipo
        self.coords = coords
        self.cor = cor
        self.preenchido = preenchido


class Objeto3D(Objeto):
    def __init__(
        self,
        nome: str,
        vertices: List[Ponto3D],
        arestas: List[Tuple[int, int]],
        cor: str = "black",
    ):
        super().__init__(nome, TipoObjeto.OBJETO_3D.value, [], cor)
        self.vertices = vertices
        self.arestas = arestas
        self.transformacoes3d = Transformacoes3D()

    def aplicar_transformacao_3d(self, matriz):
        for vertice in self.vertices:
            vertice.aplicar_transformacao(matriz)

    def get_centro_objeto_3d(self):
        if not self.vertices:
            return (0, 0, 0)
        x = sum(v.x for v in self.vertices) / len(self.vertices)
        y = sum(v.y for v in self.vertices) / len(self.vertices)
        z = sum(v.z for v in self.vertices) / len(self.vertices)
        return (x, y, z)


class SuperficieBezier(Objeto):
    M_BEZIER = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])

    def __init__(self, nome: str, pontos_controle: List[Ponto3D], cor: str = "black"):
        if len(pontos_controle) != 16:
            raise ValueError(
                "Uma superfície de Bézier requer exatamente 16 pontos de controle."
            )

        super().__init__(nome, TipoObjeto.SUPERFICIE_BEZIER.value, [], cor)
        self.pontos_controle = np.array(pontos_controle).reshape(4, 4)

    def gerar_malha(
        self, num_passos_s=10, num_passos_t=10
    ) -> List[Tuple[Ponto3D, Ponto3D]]:
        pontos_superficie = np.full(
            (num_passos_s + 1, num_passos_t + 1), None, dtype=object
        )

        Gx = np.array([[p.x for p in row] for row in self.pontos_controle])
        Gy = np.array([[p.y for p in row] for row in self.pontos_controle])
        Gz = np.array([[p.z for p in row] for row in self.pontos_controle])

        for i, s in enumerate(np.linspace(0, 1, num_passos_s + 1)):
            for j, t in enumerate(np.linspace(0, 1, num_passos_t + 1)):
                S_vec = np.array([s**3, s**2, s, 1])
                T_vec = np.array([t**3, t**2, t, 1]).reshape(4, 1)

                x = S_vec @ self.M_BEZIER @ Gx @ self.M_BEZIER.T @ T_vec
                y = S_vec @ self.M_BEZIER @ Gy @ self.M_BEZIER.T @ T_vec
                z = S_vec @ self.M_BEZIER @ Gz @ self.M_BEZIER.T @ T_vec

                pontos_superficie[i, j] = Ponto3D(x[0], y[0], z[0])

        arestas = []
        for i in range(num_passos_s + 1):
            for j in range(num_passos_t):
                arestas.append((pontos_superficie[i, j], pontos_superficie[i, j + 1]))

        for j in range(num_passos_t + 1):
            for i in range(num_passos_s):
                arestas.append((pontos_superficie[i, j], pontos_superficie[i + 1, j]))

        return arestas

    def aplicar_transformacao_3d(self, matriz):
        for i in range(4):
            for j in range(4):
                self.pontos_controle[i, j].aplicar_transformacao(matriz)


class SuperficieBSpline(Objeto):
    M_BSPLINE = (1 / 6) * np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]
    )
    M_BSPLINE_T = M_BSPLINE.T

    def __init__(
        self,
        nome: str,
        pontos_controle_matriz: List[List[Ponto3D]],
        cor: str = "black",
    ):
        """
        Inicializa a superfície B-Spline.
        pontos_controle_matriz: Uma matriz (lista de listas ou np.array)
                                de Ponto3D com dimensão N x M (mínimo 4x4).
        """
        super().__init__(nome, TipoObjeto.SUPERFICIE_BSPLINE.value, [], cor)

        if not isinstance(pontos_controle_matriz, np.ndarray):
            pontos_controle_matriz = np.array(pontos_controle_matriz)

        if pontos_controle_matriz.ndim != 2:
            raise ValueError("Pontos de controle devem ser uma matriz 2D.")

        self.n_linhas, self.n_cols = pontos_controle_matriz.shape

        if self.n_linhas < 4 or self.n_cols < 4:
            raise ValueError("A matriz de pontos de controle deve ser no mínimo 4x4.")

        self.pontos_controle_matriz = pontos_controle_matriz

    @staticmethod
    def _get_matriz_E(delta: float) -> np.ndarray:
        """Matriz de base para forward differences (igual à da curva 2D)"""
        d2 = delta * delta
        d3 = d2 * delta
        return np.array(
            [
                [0, 0, 0, 1],
                [d3, d2, delta, 0],
                [6 * d3, 2 * d2, 0, 0],
                [6 * d3, 0, 0, 0],
            ]
        )

    def _gerar_patch(
        self,
        G_patch: np.ndarray,
        num_passos_s: int,
        num_passos_t: int,
    ) -> List[Tuple[Ponto3D, Ponto3D]]:
        """
        Gera a malha para um único patch 4x4 de pontos de controle
        usando o método de Forward Differences.
        """
        Gx = np.array([[p.x for p in row] for row in G_patch])
        Gy = np.array([[p.y for p in row] for row in G_patch])
        Gz = np.array([[p.z for p in row] for row in G_patch])

        # 1. Calcular matrizes de coeficientes
        Cx = self.M_BSPLINE @ Gx @ self.M_BSPLINE_T
        Cy = self.M_BSPLINE @ Gy @ self.M_BSPLINE_T
        Cz = self.M_BSPLINE @ Gz @ self.M_BSPLINE_T

        delta_s = 1.0 / num_passos_s
        delta_t = 1.0 / num_passos_t

        Es = self._get_matriz_E(delta_s)
        Et = self._get_matriz_E(delta_t)
        Et_T = Et.T

        # 2. Calcular matrizes de diferenças
        Dx = Es @ Cx @ Et_T
        Dy = Es @ Cy @ Et_T
        Dz = Es @ Cz @ Et_T

        # 3. Iterar usando forward differences
        # (Transpomos D para que as diferenças em 't' estejam nas colunas,
        # facilitando o loop interno)
        DDx, DDy, DDz = Dx.T, Dy.T, Dz.T

        pontos_superficie = np.full(
            (num_passos_s + 1, num_passos_t + 1), None, dtype=object
        )

        # Valores iniciais e diferenças para as curvas na direção S
        curva_s_x, curva_s_y, curva_s_z = (
            DDx[0].copy(),
            DDy[0].copy(),
            DDz[0].copy(),
        )
        delta_s_x, delta_s_y, delta_s_z = (
            DDx[1].copy(),
            DDy[1].copy(),
            DDz[1].copy(),
        )
        delta_s2_x, delta_s2_y, delta_s2_z = (
            DDx[2].copy(),
            DDy[2].copy(),
            DDz[2].copy(),
        )
        delta_s3_x, delta_s3_y, delta_s3_z = (
            DDx[3].copy(),
            DDy[3].copy(),
            DDz[3].copy(),
        )

        for i in range(num_passos_s + 1):
            # Valores iniciais e diferenças para a curva atual na direção T
            x, dx, d2x, d3x = (
                curva_s_x[0],
                curva_s_x[1],
                curva_s_x[2],
                curva_s_x[3],
            )
            y, dy, d2y, d3y = (
                curva_s_y[0],
                curva_s_y[1],
                curva_s_y[2],
                curva_s_y[3],
            )
            z, dz, d2z, d3z = (
                curva_s_z[0],
                curva_s_z[1],
                curva_s_z[2],
                curva_s_z[3],
            )

            for j in range(num_passos_t + 1):
                pontos_superficie[i, j] = Ponto3D(x, y, z)
                # Avançar na direção T
                x += dx
                dx += d2x
                d2x += d3x
                y += dy
                dy += d2y
                d2y += d3y
                z += dz
                dz += d2z
                d2z += d3z

            # Avançar para a próxima curva na direção S
            curva_s_x += delta_s_x
            delta_s_x += delta_s2_x
            delta_s2_x += delta_s3_x
            curva_s_y += delta_s_y
            delta_s_y += delta_s2_y
            delta_s2_y += delta_s3_y
            curva_s_z += delta_s_z
            delta_s_z += delta_s2_z
            delta_s2_z += delta_s3_z

        # 4. Criar arestas a partir dos pontos gerados
        arestas = []
        for i in range(num_passos_s + 1):
            for j in range(num_passos_t):
                arestas.append((pontos_superficie[i, j], pontos_superficie[i, j + 1]))

        for j in range(num_passos_t + 1):
            for i in range(num_passos_s):
                arestas.append((pontos_superficie[i, j], pontos_superficie[i + 1, j]))

        return arestas

    def gerar_malha(
        self, num_passos_s=10, num_passos_t=10
    ) -> List[Tuple[Ponto3D, Ponto3D]]:
        """
        Gera a malha completa da superfície, subdividindo a matriz
        de controle em patches 4x4.
        """
        num_patches_s = self.n_linhas - 3
        num_patches_t = self.n_cols - 3
        malha_completa = []

        for i in range(num_patches_s):
            for j in range(num_patches_t):
                # Extrai o patch 4x4
                G_patch = self.pontos_controle_matriz[i : i + 4, j : j + 4]
                # Gera a malha para este patch e adiciona à lista
                malha_completa.extend(
                    self._gerar_patch(G_patch, num_passos_s, num_passos_t)
                )

        return malha_completa

    def aplicar_transformacao_3d(self, matriz):
        """Aplica uma transformação a todos os pontos de controle da superfície."""
        for i in range(self.n_linhas):
            for j in range(self.n_cols):
                self.pontos_controle_matriz[i, j].aplicar_transformacao(matriz)