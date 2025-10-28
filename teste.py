import tkinter as tk
from tkinter import (
    ttk,
    colorchooser,
    messagebox,
    simpledialog,
    filedialog,
)  # Adicionado filedialog
from typing import List, Tuple
from enum import Enum
import numpy as np
import ast
import copy


class Ponto3D:
    """Representa um ponto em coordenadas homogêneas 3D (x, y, z, w)."""

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
        """Aplica uma matriz de transformação 4x4 ao ponto."""
        self.coords = matriz @ self.coords


class Transformacoes3D:
    """Fornece matrizes de transformação 3D."""

    @staticmethod
    def get_matriz_translacao(dx, dy, dz):
        return np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])

    @staticmethod
    def get_matriz_escalonamento(sx, sy, sz):
        return np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])

    @staticmethod
    def get_matriz_rotacao_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def get_matriz_rotacao_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    @staticmethod
    def get_matriz_rotacao_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    @staticmethod
    def _normalizar(v):
        n = np.linalg.norm(v)
        return v / n if n != 0 else v

    @classmethod
    def get_matriz_rotacao_eixo_arbitrario(cls, p1, p2, a):
        e = cls._normalizar(np.array(p2) - np.array(p1))
        ax, ay, az = e
        d = np.sqrt(ay**2 + az**2)
        T = cls.get_matriz_translacao(-p1[0], -p1[1], -p1[2])
        Ti = cls.get_matriz_translacao(p1[0], p1[1], p1[2])
        Rx = np.identity(4)
        Rxi = Rx.T
        if d != 0:
            Rx = np.array(
                [
                    [1, 0, 0, 0],
                    [0, az / d, -ay / d, 0],
                    [0, ay / d, az / d, 0],
                    [0, 0, 0, 1],
                ]
            )
            Rxi = Rx.T
        Ry = np.array([[d, 0, -ax, 0], [0, 1, 0, 0], [ax, 0, d, 0], [0, 0, 0, 1]])
        Ryi = Ry.T
        Rz = cls.get_matriz_rotacao_z(a)
        return Ti @ Rxi @ Ryi @ Rz @ Ry @ Rx @ T


class Camera:
    """Gerencia a visualização 3D (VRP, VPN, VUP)."""

    def __init__(self, vrp: Ponto3D, p_foco: Ponto3D, vup: tuple, d: float = 300):
        self.vrp, self.p_foco, self.vup, self.d = vrp, p_foco, np.array(vup), d
        self.matriz_wc = self._calcular_matriz_wc()

    def set_distancia_plano_projecao(self, d):
        self.d = d

    def _calcular_matriz_wc(self):
        vpn = self.p_foco.coords[:3] - self.vrp.coords[:3]
        n = Transformacoes3D._normalizar(vpn)
        u = Transformacoes3D._normalizar(np.cross(self.vup, n))
        v = Transformacoes3D._normalizar(np.cross(n, u))
        mt = np.array(
            [
                [1, 0, 0, -self.vrp.x],
                [0, 1, 0, -self.vrp.y],
                [0, 0, 1, -self.vrp.z],
                [0, 0, 0, 1],
            ]
        )
        mr = np.array([np.append(u, 0), np.append(v, 0), np.append(n, 0), [0, 0, 0, 1]])
        return mr @ mt

    def get_matriz_wc(self):
        return self.matriz_wc

    def mover(self, dx, dy, dz):  # Movimento absoluto nos eixos do mundo
        m = np.array([dx, dy, dz, 0])
        self.vrp.coords += m
        self.p_foco.coords += m
        self.matriz_wc = self._calcular_matriz_wc()

    def girar(self, ay, ax):  # Yaw (eixo Y mundo), Pitch (eixo u camera)
        vpn = self.p_foco.coords[:3] - self.vrp.coords[:3]
        n = Transformacoes3D._normalizar(vpn)
        u = Transformacoes3D._normalizar(np.cross(self.vup, n))
        ry = Transformacoes3D.get_matriz_rotacao_y(ay)
        rx = Transformacoes3D.get_matriz_rotacao_eixo_arbitrario((0, 0, 0), u, ax)
        nvpn = (ry @ rx @ np.append(vpn, 1))[:3]
        self.p_foco.coords[:3] = self.vrp.coords[:3] + nvpn
        self.matriz_wc = self._calcular_matriz_wc()

    def mover_relativo(
        self, dr, dc, df
    ):  # Movimento relativo aos eixos da camera (direita, cima, frente)
        vpn = self.p_foco.coords[:3] - self.vrp.coords[:3]
        n = Transformacoes3D._normalizar(vpn)
        u = Transformacoes3D._normalizar(np.cross(self.vup, n))
        v = Transformacoes3D._normalizar(np.cross(n, u))
        mw = (dr * u) + (dc * v) + (df * n)
        m = Transformacoes3D.get_matriz_translacao(mw[0], mw[1], mw[2])
        self.vrp.aplicar_transformacao(m)
        self.p_foco.aplicar_transformacao(m)
        self.matriz_wc = self._calcular_matriz_wc()


class Transformacoes:  # Mantida para objetos 2D
    @staticmethod
    def get_matriz_translacao(dx, dy):
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    @staticmethod
    def get_matriz_escalonamento(sx, sy):
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    @staticmethod
    def get_matriz_rotacao(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def aplicar_transformacao_generica(self, obj, m):
        obj.coords = [tuple(m @ np.array([x, y, 1])[:2]) for x, y in obj.coords]

    def get_centro_objeto(self, obj):
        if not obj.coords:
            return (0, 0)
        return np.mean(obj.coords, axis=0)

    def aplicar_escalonamento_natural(self, obj, sx, sy):
        cx, cy = self.get_centro_objeto(obj)
        m = (
            self.get_matriz_translacao(cx, cy)
            @ self.get_matriz_escalonamento(sx, sy)
            @ self.get_matriz_translacao(-cx, -cy)
        )
        self.aplicar_transformacao_generica(obj, m)

    def aplicar_rotacao_centro_objeto(self, obj, a):
        cx, cy = self.get_centro_objeto(obj)
        m = (
            self.get_matriz_translacao(cx, cy)
            @ self.get_matriz_rotacao(a)
            @ self.get_matriz_translacao(-cx, -cy)
        )
        self.aplicar_transformacao_generica(obj, m)

    def aplicar_rotacao_ponto_arbitrario(self, obj, a, px, py):
        m = (
            self.get_matriz_translacao(px, py)
            @ self.get_matriz_rotacao(a)
            @ self.get_matriz_translacao(-px, -py)
        )
        self.aplicar_transformacao_generica(obj, m)


class TipoObjeto(Enum):
    (
        PONTO,
        RETA,
        POLIGONO,
        CURVA,
        B_SPLINE,
        OBJETO_3D,
        SUPERFICIE_BEZIER,
        SUPERFICIE_BSPLINE,
    ) = (
        "ponto",
        "reta",
        "poligono",
        "curva",
        "b-spline",
        "objeto_3d",
        "superficie_bezier",
        "superficie_bspline",
    )


class Objeto:
    def __init__(
        self,
        nome: str,
        tipo: str,
        coords: List,
        cor: str = "black",
        preenchido: bool = False,
    ):
        self.nome, self.tipo, self.coords, self.cor, self.preenchido = (
            nome,
            tipo,
            coords,
            cor,
            preenchido,
        )


class Objeto3D(Objeto):
    def __init__(
        self,
        nome: str,
        vertices: List[Ponto3D],
        arestas: List[Tuple[int, int]],
        cor: str = "black",
    ):
        super().__init__(nome, TipoObjeto.OBJETO_3D.value, [], cor)
        self.vertices, self.arestas = vertices, arestas

    def aplicar_transformacao_3d(self, m):
        [v.aplicar_transformacao(m) for v in self.vertices]

    def get_centro_objeto_3d(self):
        return (
            np.mean([v.coords[:3] for v in self.vertices], axis=0)
            if self.vertices
            else (0, 0, 0)
        )


class SuperficieBezier(Objeto):
    M_BEZIER = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])

    def __init__(self, nome: str, p_ctrl: List[Ponto3D], cor: str = "black"):
        if len(p_ctrl) != 16:
            raise ValueError("Superfície Bézier requer 16 pontos.")
        super().__init__(nome, TipoObjeto.SUPERFICIE_BEZIER.value, [], cor)
        self.pontos_controle = np.array(p_ctrl).reshape(4, 4)

    def gerar_malha(self, ns=10, nt=10) -> List[Tuple[Ponto3D, Ponto3D]]:
        pts = np.full((ns + 1, nt + 1), None, dtype=object)
        Gx = np.array([[p.x for p in r] for r in self.pontos_controle])
        Gy = np.array([[p.y for p in r] for r in self.pontos_controle])
        Gz = np.array([[p.z for p in r] for r in self.pontos_controle])
        for i, s in enumerate(np.linspace(0, 1, ns + 1)):
            for j, t in enumerate(np.linspace(0, 1, nt + 1)):
                S = np.array([s**3, s**2, s, 1])
                T = np.array([t**3, t**2, t, 1]).reshape(4, 1)
                x = (S @ self.M_BEZIER @ Gx @ self.M_BEZIER.T @ T)[0]
                y = (S @ self.M_BEZIER @ Gy @ self.M_BEZIER.T @ T)[0]
                z = (S @ self.M_BEZIER @ Gz @ self.M_BEZIER.T @ T)[0]
                pts[i, j] = Ponto3D(x, y, z)
        arestas = []
        for i in range(ns + 1):
            for j in range(nt):
                arestas.append((pts[i, j], pts[i, j + 1]))
        for j in range(nt + 1):
            for i in range(ns):
                arestas.append((pts[i, j], pts[i + 1, j]))
        return arestas

    def aplicar_transformacao_3d(self, m):
        [[p.aplicar_transformacao(m) for p in r] for r in self.pontos_controle]


class SuperficieBSpline(Objeto):
    M_BSPLINE = (1 / 6) * np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]
    )

    def __init__(
        self,
        nome: str,
        pontos_controle_grid: List[List[Ponto3D]],
        ns: int,
        nt: int,
        cor: str = "black",
    ):
        super().__init__(nome, TipoObjeto.SUPERFICIE_BSPLINE.value, [], cor)
        self.grid = np.array(pontos_controle_grid)
        self.rows, self.cols = self.grid.shape
        if self.rows < 4 or self.cols < 4:
            raise ValueError("Grid >= 4x4.")
        self.ns, self.nt = ns, nt

    def _get_patch(self, r, c):
        return self.grid[r : r + 4, c : c + 4]

    def _get_matriz_E(self, delta):
        d2, d3 = delta**2, delta**3
        return np.array(
            [
                [0, 0, 0, 1],
                [d3, d2, delta, 0],
                [6 * d3, 2 * d2, 0, 0],
                [6 * d3, 0, 0, 0],
            ]
        )

    def _calcular_curva_fwd_diff(self, n_passos, f0, df0, d2f0, d3f0):
        pts = [np.array(f0)]
        x, y, z = f0
        dx, dy, dz = df0
        d2x, d2y, d2z = d2f0
        d3x, d3y, d3z = d3f0
        for _ in range(n_passos - 1):
            x += dx
            dx += d2x
            d2x += d3x
            y += dy
            dy += d2y
            d2y += d3y
            z += dz
            dz += d2z
            d2z += d3z
            pts.append(np.array([x, y, z]))
        return pts

    def _calcular_patch_fwd_diff(self, patch, ns, nt):
        if ns < 2 or nt < 2:
            return []
        ds, dt = 1 / (ns - 1), 1 / (nt - 1)
        Es, Et = self._get_matriz_E(ds), self._get_matriz_E(dt)
        Gx = np.array([[p.x for p in r] for r in patch])
        Gy = np.array([[p.y for p in r] for r in patch])
        Gz = np.array([[p.z for p in r] for r in patch])
        Cx = self.M_BSPLINE @ Gx @ self.M_BSPLINE.T
        Cy = self.M_BSPLINE @ Gy @ self.M_BSPLINE.T
        Cz = self.M_BSPLINE @ Gz @ self.M_BSPLINE.T
        DDx = Es @ Cx @ Et.T
        DDy = Es @ Cy @ Et.T
        DDz = Es @ Cz @ Et.T
        segments = []
        DDx_o, DDy_o, DDz_o = DDx.copy(), DDy.copy(), DDz.copy()
        # Curvas em t
        for _ in range(ns):
            f0x, df0x, d2f0x, d3f0x = DDx[0, 0], DDx[0, 1], DDx[0, 2], DDx[0, 3]
            f0y, df0y, d2f0y, d3f0y = DDy[0, 0], DDy[0, 1], DDy[0, 2], DDy[0, 3]
            f0z, df0z, d2f0z, d3f0z = DDz[0, 0], DDz[0, 1], DDz[0, 2], DDz[0, 3]
            pts = self._calcular_curva_fwd_diff(
                nt,
                (f0x, f0y, f0z),
                (df0x, df0y, df0z),
                (d2f0x, d2f0y, d2f0z),
                (d3f0x, d3f0y, d3f0z),
            )
            for k in range(len(pts) - 1):
                segments.append((Ponto3D(*pts[k]), Ponto3D(*pts[k + 1])))
            DDx[0] += DDx[1]
            DDx[1] += DDx[2]
            DDx[2] += DDx[3]
            DDy[0] += DDy[1]
            DDy[1] += DDy[2]
            DDy[2] += DDy[3]
            DDz[0] += DDz[1]
            DDz[1] += DDz[2]
            DDz[2] += DDz[3]
        # Curvas em s
        DDx, DDy, DDz = DDx_o.T, DDy_o.T, DDz_o.T  # Reset e Transpõe
        for _ in range(nt):
            f0x, df0x, d2f0x, d3f0x = DDx[0, 0], DDx[0, 1], DDx[0, 2], DDx[0, 3]
            f0y, df0y, d2f0y, d3f0y = DDy[0, 0], DDy[0, 1], DDy[0, 2], DDy[0, 3]
            f0z, df0z, d2f0z, d3f0z = DDz[0, 0], DDz[0, 1], DDz[0, 2], DDz[0, 3]
            pts = self._calcular_curva_fwd_diff(
                ns,
                (f0x, f0y, f0z),
                (df0x, df0y, df0z),
                (d2f0x, d2f0y, d2f0z),
                (d3f0x, d3f0y, d3f0z),
            )
            for k in range(len(pts) - 1):
                segments.append((Ponto3D(*pts[k]), Ponto3D(*pts[k + 1])))
            DDx[0] += DDx[1]
            DDx[1] += DDx[2]
            DDx[2] += DDx[3]
            DDy[0] += DDy[1]
            DDy[1] += DDy[2]
            DDy[2] += DDy[3]
            DDz[0] += DDz[1]
            DDz[1] += DDz[2]
            DDz[2] += DDz[3]
        return segments

    def gerar_malha_fwd_diff(self):
        all_segments = []
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                all_segments.extend(
                    self._calcular_patch_fwd_diff(
                        self._get_patch(r, c), self.ns, self.nt
                    )
                )
        return all_segments

    def aplicar_transformacao_3d(self, m):
        [[p.aplicar_transformacao(m) for p in r] for r in self.grid]


class DisplayFile:
    def __init__(self):
        self.objetos = []

    def adicionar(self, o):
        self.objetos.append(o)

    def listar(self):
        return self.objetos

    def get_by_name(self, n):
        return next((o for o in self.objetos if o.nome == n), None)


class GeradorBSpline:  # Para curvas 2D B-Spline
    M_BSPLINE = (1 / 6) * np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]
    )

    @staticmethod
    def _get_matriz_E(d):
        d2, d3 = d * d, d * d * d
        return np.array(
            [[0, 0, 0, 1], [d3, d2, d, 0], [6 * d3, 2 * d2, 0, 0], [6 * d3, 0, 0, 0]]
        )

    def gerar_segmentos(self, p_c, delta=0.1):
        if len(p_c) < 4:
            return []
        segs, n_curves = [], len(p_c) - 3
        E, n_steps = self._get_matriz_E(delta), int(1 / delta)
        for i in range(n_curves):
            Gx, Gy = np.array([p[0] for p in p_c[i : i + 4]]).reshape(4, 1), np.array(
                [p[1] for p in p_c[i : i + 4]]
            ).reshape(4, 1)
            Cx, Cy = self.M_BSPLINE @ Gx, self.M_BSPLINE @ Gy
            Dx, Dy = (E @ Cx).flatten(), (E @ Cy).flatten()
            x, dx, d2x, d3x = Dx[0], Dx[1], Dx[2], Dx[3]
            y, dy, d2y, d3y = Dy[0], Dy[1], Dy[2], Dy[3]
            for _ in range(n_steps):
                xo, yo = x, y
                x += dx
                dx += d2x
                d2x += d3x
                y += dy
                dy += d2y
                d2y += d3y
                segs.append([(xo, yo), (x, y)])
        return segs


class Clipping:
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def _get_outcode(self, x, y, wmin, wmax):
        c = self.INSIDE
        c |= self.LEFT if x < wmin[0] else (self.RIGHT if x > wmax[0] else 0)
        c |= self.BOTTOM if y < wmin[1] else (self.TOP if y > wmax[1] else 0)
        return c

    def cohen_sutherland(self, p1, p2, wmin, wmax):
        x1, y1 = p1
        x2, y2 = p2
        c1, c2 = self._get_outcode(x1, y1, wmin, wmax), self._get_outcode(
            x2, y2, wmin, wmax
        )
        while True:
            if not (c1 | c2):
                return [(x1, y1), (x2, y2)]
            elif c1 & c2:
                return []
            else:
                c_out = c1 if c1 else c2
                x, y = 0, 0
                try:  # Tratamento de divisão por zero
                    if c_out & self.TOP:
                        x, y = (
                            (x1 + (x2 - x1) * (wmax[1] - y1) / (y2 - y1))
                            if y2 != y1
                            else x1
                        ), wmax[1]
                    elif c_out & self.BOTTOM:
                        x, y = (
                            (x1 + (x2 - x1) * (wmin[1] - y1) / (y2 - y1))
                            if y2 != y1
                            else x1
                        ), wmin[1]
                    elif c_out & self.RIGHT:
                        y, x = (
                            (y1 + (y2 - y1) * (wmax[0] - x1) / (x2 - x1))
                            if x2 != x1
                            else y1
                        ), wmax[0]
                    elif c_out & self.LEFT:
                        y, x = (
                            (y1 + (y2 - y1) * (wmin[0] - x1) / (x2 - x1))
                            if x2 != x1
                            else y1
                        ), wmin[0]
                except ZeroDivisionError:
                    return []  # Reta horizontal/vertical fora
                if c_out == c1:
                    x1, y1 = x, y
                    c1 = self._get_outcode(x1, y1, wmin, wmax)
                else:
                    x2, y2 = x, y
                    c2 = self._get_outcode(x2, y2, wmin, wmax)

    def liang_barsky(self, p1, p2, wmin, wmax):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        p, q = [-dx, dx, -dy, dy], [
            x1 - wmin[0],
            wmax[0] - x1,
            y1 - wmin[1],
            wmax[1] - y1,
        ]
        u1, u2 = 0.0, 1.0
        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return []
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    u1 = max(u1, t)
                else:
                    u2 = min(u2, t)
        if u1 > u2:
            return []
        return [(x1 + u1 * dx, y1 + u1 * dy), (x1 + u2 * dx, y1 + u2 * dy)]

    def sutherland_hodgman(self, v, wmin, wmax):
        def clip(v_in, edge):
            v_out = []
            for i in range(len(v_in)):
                p1, p2 = v_in[i], v_in[(i + 1) % len(v_in)]
                in1, in2 = is_inside(p1, edge), is_inside(p2, edge)
                if in1 and in2:
                    v_out.append(p2)
                elif in1 and not in2:
                    v_out.append(intersection(p1, p2, edge))
                elif not in1 and in2:
                    v_out.append(intersection(p1, p2, edge))
                    v_out.append(p2)
            return v_out

        def is_inside(p, edge):
            if edge == "l":
                return p[0] >= wmin[0]
            if edge == "r":
                return p[0] <= wmax[0]
            if edge == "b":
                return p[1] >= wmin[1]
            if edge == "t":
                return p[1] <= wmax[1]

        def intersection(p1, p2, edge):
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1
            try:
                if edge == "l":
                    y = y1 + dy * (wmin[0] - x1) / dx if dx != 0 else y1
                    return (wmin[0], y)
                if edge == "r":
                    y = y1 + dy * (wmax[0] - x1) / dx if dx != 0 else y1
                    return (wmax[0], y)
                if edge == "b":
                    x = x1 + dx * (wmin[1] - y1) / dy if dy != 0 else x1
                    return (x, wmin[1])
                if edge == "t":
                    x = x1 + dx * (wmax[1] - y1) / dy if dy != 0 else x1
                    return (x, wmax[1])
            except ZeroDivisionError:
                return p1

        out_v = v
        for e in ["l", "r", "b", "t"]:
            if not out_v:
                break
                out_v = clip(out_v, e)
        return out_v

    def _gen_bezier_pts(self, p, n=20):
        return [
            tuple(
                (1 - t) ** 3 * np.array(p[0])
                + 3 * (1 - t) ** 2 * t * np.array(p[1])
                + 3 * (1 - t) * t**2 * np.array(p[2])
                + t**3 * np.array(p[3])
            )
            for t in (i / n for i in range(n + 1))
        ]

    def _subdiv_bezier(self, p, t):
        p1, p2, p3, p4 = p
        p12, p23, p34 = (
            (1 - t) * np.array(p1) + t * np.array(p2),
            (1 - t) * np.array(p2) + t * np.array(p3),
            (1 - t) * np.array(p3) + t * np.array(p4),
        )
        p123, p234 = (1 - t) * p12 + t * p23, (1 - t) * p23 + t * p34
        p1234 = (1 - t) * p123 + t * p234
        return [p1, tuple(p12), tuple(p123), tuple(p1234)], [
            tuple(p1234),
            tuple(p234),
            tuple(p34),
            p4,
        ]

    def _clip_bezier_rec(self, p, wmin, wmax, prof=0):
        if prof > 10:
            l = self.cohen_sutherland(p[0], p[-1], wmin, wmax)
            return [l] if l else []
        if all(self._get_outcode(pt[0], pt[1], wmin, wmax) == self.INSIDE for pt in p):
            pts = self._gen_bezier_pts(p)
            return [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
        oc = 0b1111
        [oc := oc & self._get_outcode(pt[0], pt[1], wmin, wmax) for pt in p]
        if oc != self.INSIDE:
            return []
        c1, c2 = self._subdiv_bezier(p, 0.5)
        segs = []
        segs.extend(self._clip_bezier_rec(c1, wmin, wmax, prof + 1))
        segs.extend(self._clip_bezier_rec(c2, wmin, wmax, prof + 1))
        return segs

    def clip_curva_para_linhas(self, p_total, wmin, wmax):
        if len(p_total) < 4:
            return []
        # CORREÇÃO: Mover esta linha para fora do if e antes do for loop
        segs, n_segs = [], (len(p_total) - 4) // 3 + 1
        for i in range(n_segs):
            segs.extend(self._clip_bezier_rec(p_total[i * 3 : i * 3 + 4], wmin, wmax))
        return segs

    def clip(self, obj, wmin, wmax, alg):
        coords = []
        t = obj.tipo
        if t == TipoObjeto.PONTO.value:
            x, y = obj.coords[0]
            coords = (
                obj.coords
                if wmin[0] <= x <= wmax[0] and wmin[1] <= y <= wmax[1]
                else []
            )
        elif t == TipoObjeto.RETA.value:
            coords = (
                self.cohen_sutherland(obj.coords[0], obj.coords[1], wmin, wmax)
                if alg == "cs"
                else self.liang_barsky(obj.coords[0], obj.coords[1], wmin, wmax)
            )
        elif t == TipoObjeto.POLIGONO.value:
            coords = self.sutherland_hodgman(obj.coords, wmin, wmax)
        # Retorna o objeto original para tipos complexos, o clipping 2D será feito nos segmentos gerados
        if t in [
            TipoObjeto.CURVA.value,
            TipoObjeto.B_SPLINE.value,
            TipoObjeto.OBJETO_3D.value,
            TipoObjeto.SUPERFICIE_BEZIER.value,
            TipoObjeto.SUPERFICIE_BSPLINE.value,
        ]:
            return obj
        # Se coordenadas ficaram vazias após clipping simples, retorna None
        if not coords:
            return None
        # Cria novo objeto com coordenadas clipadas para tipos simples
        return Objeto(obj.nome, t, coords, obj.cor, obj.preenchido)


class Viewport:
    def __init__(self, canvas, wmin, wmax, camera):
        self.canvas, self.wmin, self.wmax, self.angle = canvas, wmin, wmax, 0.0
        self.clip, self.bsp_gen, self.cam = Clipping(), GeradorBSpline(), camera

    def set_window_angle(self, a):
        self.angle = a

    def _rotate_pt(self, x, y, cx, cy, a):
        tx, ty = x - cx, y - cy
        c, s = np.cos(a), np.sin(a)
        return c * tx - s * ty + cx, s * tx + c * ty + cy

    def transform(self, xw, yw, wmin, wmax, vpmin, vpmax):
        ww, wh = wmax[0] - wmin[0], wmax[1] - wmin[1]
        vpw, vph = vpmax[0] - vpmin[0], vpmax[1] - vpmin[1]
        if ww == 0 or wh == 0:
            return (0, 0)
        xvp = vpmin[0] + ((xw - wmin[0]) / ww) * vpw
        yvp = vpmin[1] + (1 - ((yw - wmin[1]) / wh)) * vph
        return (xvp, yvp)

    def desenhar(self, df: DisplayFile, clip_alg: str, proj_type: str):
        self.canvas.delete("all")
        cw, ch, m = self.canvas.winfo_width(), self.canvas.winfo_height(), 10
        if cw <= 2 * m or ch <= 2 * m:
            return
        aw, ah = cw - 2 * m, ch - 2 * m
        ww, wh = self.wmax[0] - self.wmin[0], self.wmax[1] - self.wmin[1]
        w_asp, vp_asp = ww / wh if wh != 0 else 1, aw / ah
        if vp_asp > w_asp:
            vph = ah
            vpw = vph * w_asp
            ox, oy = m + (aw - vpw) / 2, m
        else:
            vpw = aw
            vph = vpw / w_asp
            ox, oy = m, m + (ah - vph) / 2
        vpmin, vpmax = (ox, oy), (ox + vpw, oy + vph)
        self.canvas.create_rectangle(
            vpmin[0], vpmin[1], vpmax[0], vpmax[1], outline="blue", dash=(4, 4)
        )
        cx, cy, ang = (
            (self.wmin[0] + self.wmax[0]) / 2,
            (self.wmin[1] + self.wmax[1]) / 2,
            -np.deg2rad(self.angle),
        )

        def tf(xw, yw):
            xr, yr = self._rotate_pt(xw, yw, cx, cy, ang)
            return self.transform(xr, yr, self.wmin, self.wmax, vpmin, vpmax)

        ovp = tf(0, 0)
        psx, pex = tf(self.wmin[0], 0), tf(self.wmax[0], 0)
        psy, pey = tf(0, self.wmin[1]), tf(0, self.wmax[1])
        self.canvas.create_line(
            psx[0], ovp[1], pex[0], ovp[1], fill="gray", dash=(2, 2)
        )
        self.canvas.create_line(
            ovp[0], psy[1], ovp[0], pey[1], fill="gray", dash=(2, 2)
        )

        for obj in df.listar():
            # --- Objetos 3D e Superfícies ---
            if isinstance(
                obj, (Objeto3D, SuperficieBezier, SuperficieBSpline)
            ):  # ADIÇÃO: Inclui SuperficieBSpline
                obj_cam = copy.deepcopy(obj)
                obj_cam.aplicar_transformacao_3d(self.cam.get_matriz_wc())
                segments, width = [], 1
                if isinstance(obj_cam, Objeto3D):
                    segments = [
                        (obj_cam.vertices[i1], obj_cam.vertices[i2])
                        for i1, i2 in obj_cam.arestas
                    ]
                    width = 2
                elif isinstance(obj_cam, SuperficieBezier):
                    segments = obj_cam.gerar_malha()
                elif isinstance(obj_cam, SuperficieBSpline):
                    segments = (
                        obj_cam.gerar_malha_fwd_diff()
                    )  # ADIÇÃO: Chama método fwd_diff

                for p1c, p2c in segments:
                    p1_2d, p2_2d = (0, 0), (0, 0)  # Inicializa
                    # Projeção
                    if proj_type == "perspectiva":
                        d = self.cam.d
                        near = 0.1  # Plano near um pouco a frente da câmera
                        z1, z2 = p1c.z, p2c.z

                        # Clipping 3D Básico (Plano Near) - Simplificado: descarta segmento se qualquer ponto estiver atrás ou muito perto
                        if z1 >= -near or z2 >= -near:
                            continue

                        # Divisão perspectiva (Câmera olhando para -Z)
                        # Verifica se z1 ou z2 são zero ou muito próximos de zero antes da divisão
                        if abs(z1) < 1e-6 or abs(z2) < 1e-6:
                            continue  # Evita divisão por zero/valor muito pequeno

                        p1_2d = (-d * p1c.x / z1, -d * p1c.y / z1)
                        p2_2d = (-d * p2c.x / z2, -d * p2c.y / z2)
                    else:  # Ortogonal
                        p1_2d, p2_2d = (p1c.x, p1c.y), (p2c.x, p2c.y)

                    # Clipping 2D e Desenho
                    clipped = self.clip.cohen_sutherland(
                        p1_2d, p2_2d, self.wmin, self.wmax
                    )  # Usa CS para todos segmentos 3D projetados
                    if clipped:
                        p1vp, p2vp = tf(*clipped[0]), tf(*clipped[1])
                        self.canvas.create_line(p1vp, p2vp, fill=obj.cor, width=width)
                continue  # Próximo objeto na lista

            # --- Processamento de Objetos 2D (mantido como no original) ---
            # CORREÇÃO: Usar obj.tipo (string) na comparação
            if obj.tipo == TipoObjeto.CURVA.value:
                lines = self.clip.clip_curva_para_linhas(
                    obj.coords, self.wmin, self.wmax
                )
                [
                    self.canvas.create_line(tf(*l[0]), tf(*l[1]), fill=obj.cor, width=2)
                    for l in lines
                    if l
                ]
                continue
            if obj.tipo == TipoObjeto.B_SPLINE.value:
                lines = self.bsp_gen.gerar_segmentos(obj.coords)
                [
                    (
                        lambda c: (
                            self.canvas.create_line(
                                tf(*c.coords[0]),
                                tf(*c.coords[1]),
                                fill=obj.cor,
                                width=2,
                            )
                            if c
                            else None
                        )
                    )(
                        self.clip.clip(
                            Objeto("", "reta", [p1, p2]), self.wmin, self.wmax, clip_alg
                        )
                    )
                    for p1, p2 in lines
                ]
                continue
            obj_c = self.clip.clip(obj, self.wmin, self.wmax, clip_alg)
            if not obj_c:
                continue
            coords_vp = [tf(x, y) for x, y in obj_c.coords]
            # CORREÇÃO: Usar obj_c.tipo (string) na comparação
            if obj_c.tipo == TipoObjeto.PONTO.value:
                x, y = coords_vp[0]
                self.canvas.create_oval(
                    x - 2, y - 2, x + 2, y + 2, fill=obj_c.cor, outline=obj_c.cor
                )
            elif obj_c.tipo == TipoObjeto.RETA.value:
                if len(coords_vp) >= 2:
                    self.canvas.create_line(coords_vp, fill=obj_c.cor, width=2)
            elif obj_c.tipo == TipoObjeto.POLIGONO.value:
                if len(coords_vp) > 1:
                    self.canvas.create_polygon(
                        coords_vp,
                        fill=obj_c.cor if obj_c.preenchido else "",
                        outline=obj_c.cor if not obj_c.preenchido else "",
                        width=2,
                    )


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.sel_obj_name = None
        self.tf2d, self.df = Transformacoes(), DisplayFile()
        self.cam = Camera(
            vrp=Ponto3D(80, 60, 400), p_foco=Ponto3D(0, 0, 0), vup=(0, 1, 0), d=300
        )
        # --- UI Setup (Layout original mantido) ---
        fm = tk.Frame(self)
        fm.pack(fill="both", expand=True)
        fm.grid_columnconfigure(0, weight=1)
        fm.grid_rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(fm, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.vp = Viewport(
            self.canvas, wmin=(-200, -200), wmax=(200, 200), camera=self.cam
        )
        self.obj_list = tk.Listbox(fm, width=30)
        self.obj_list.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.obj_list.bind("<<ListboxSelect>>", self.on_sel)
        ctrl = tk.Frame(self)
        ctrl.pack(fill="x", padx=5, pady=(0, 5))
        tr, mr, pr, br = tk.Frame(ctrl), tk.Frame(ctrl), tk.Frame(ctrl), tk.Frame(ctrl)
        tr.pack(fill="x")
        mr.pack(fill="x", pady=2)
        pr.pack(fill="x", pady=2)
        br.pack(fill="x")

        tk.Button(tr, text="Add Obj", command=self.popup_add).pack(side="left", padx=2)
        tk.Button(tr, text="Load .obj", command=self.load_obj).pack(side="left", padx=2)
        tk.Button(tr, text="Apply TF", command=self.popup_tf).pack(side="left", padx=2)

        tk.Label(mr, text="Nav 2D:").pack(side="left", padx=(10, 0))
        [
            tk.Button(mr, text=t, command=c).pack(side="left")
            for t, c in [
                ("←", lambda: self.pan(10, 0)),
                ("→", lambda: self.pan(-10, 0)),
                ("↑", lambda: self.pan(0, -10)),
                ("↓", lambda: self.pan(0, 10)),
            ]
        ]
        tk.Button(mr, text="Zoom +", command=lambda: self.zoom(0.9)).pack(
            side="left", padx=(5, 2)
        )
        tk.Button(mr, text="Zoom -", command=lambda: self.zoom(1.1)).pack(side="left")
        tk.Button(mr, text="Rot Win", command=self.popup_rot_win).pack(
            side="left", padx=(5, 2)
        )

        tk.Label(pr, text="Nav 3D:").pack(side="left")
        [
            tk.Button(pr, text=t, command=c).pack(side="left")
            for t, c in [
                ("Frente", lambda: self.cam.mover_relativo(0, 0, 20) or self.redraw()),
                ("Trás", lambda: self.cam.mover_relativo(0, 0, -20) or self.redraw()),
                ("Esq.", lambda: self.cam.mover_relativo(-20, 0, 0) or self.redraw()),
                ("Dir.", lambda: self.cam.mover_relativo(20, 0, 0) or self.redraw()),
                ("Cima", lambda: self.cam.mover_relativo(0, 20, 0) or self.redraw()),
                ("Baixo", lambda: self.cam.mover_relativo(0, -20, 0) or self.redraw()),
            ]
        ]
        tk.Button(
            pr,
            text="⟲(Y)",
            command=lambda: self.cam.girar(np.deg2rad(5), 0) or self.redraw(),
        ).pack(side="left", padx=(5, 0))
        tk.Button(
            pr,
            text="⟳(Y)",
            command=lambda: self.cam.girar(np.deg2rad(-5), 0) or self.redraw(),
        ).pack(side="left")
        tk.Button(
            pr,
            text="⬆(P)",
            command=lambda: self.cam.girar(0, np.deg2rad(5)) or self.redraw(),
        ).pack(side="left")
        tk.Button(
            pr,
            text="⬇(P)",
            command=lambda: self.cam.girar(0, np.deg2rad(-5)) or self.redraw(),
        ).pack(side="left")

        self.proj_var = tk.StringVar(value="perspectiva")
        tk.Label(br, text="Proj:").pack(side="left")
        tk.Radiobutton(
            br,
            text="Ortogonal",
            var=self.proj_var,
            value="ortogonal",
            command=self.redraw,
        ).pack(side="left")
        tk.Radiobutton(
            br,
            text="Perspectiva",
            var=self.proj_var,
            value="perspectiva",
            command=self.redraw,
        ).pack(side="left")
        tk.Label(br, text="Dist(d):").pack(side="left", padx=(10, 0))
        self.dist_slider = tk.Scale(
            br, from_=10, to=1500, orient="h", length=150, command=self.update_dist
        )
        self.dist_slider.set(self.cam.d)
        self.dist_slider.pack(side="left")
        cf = tk.Frame(br, bd=1, relief="groove")
        cf.pack(side="left", padx=10)
        tk.Label(cf, text="Clip Reta:").pack(side="left", padx=5)
        self.clip_var = tk.StringVar(value="cs")
        tk.Radiobutton(
            cf, text="CS", var=self.clip_var, value="cs", command=self.redraw
        ).pack(side="left")
        tk.Radiobutton(
            cf, text="LB", var=self.clip_var, value="lb", command=self.redraw
        ).pack(side="left", padx=(0, 5))

        self.canvas.bind("<Configure>", self.on_resize)
        self.sel_color = "black"

    # --- Métodos ---
    def update_dist(self, v):
        self.cam.set_distancia_plano_projecao(float(v))
        self.redraw()

    def redraw(self):
        self.vp.desenhar(self.df, self.clip_var.get(), self.proj_var.get())

    def on_resize(self, e):
        self.redraw()

    def on_sel(self, e):
        sel = self.obj_list.curselection()
        self.sel_obj_name = (
            self.obj_list.get(sel[0]).rsplit(" (", 1)[0] if sel else None
        )

    def pan(self, dx, dy):
        t = np.deg2rad(self.vp.angle)
        c, s = np.cos(t), np.sin(t)
        wdx, wdy = dx * c - dy * s, dx * s + dy * c
        self.vp.wmin = (self.vp.wmin[0] + wdx, self.vp.wmin[1] + wdy)
        self.vp.wmax = (self.vp.wmax[0] + wdx, self.vp.wmax[1] + wdy)
        self.redraw()

    def zoom(self, f):
        cx, cy = (self.vp.wmin[0] + self.vp.wmax[0]) / 2, (
            self.vp.wmin[1] + self.vp.wmax[1]
        ) / 2
        w, h = (self.vp.wmax[0] - self.vp.wmin[0]) * f, (
            self.vp.wmax[1] - self.vp.wmin[1]
        ) * f
        self.vp.wmin, self.vp.wmax = (cx - w / 2, cy - h / 2), (cx + w / 2, cy + h / 2)
        self.redraw()

    def popup_rot_win(self):
        popup = tk.Toplevel(self)
        popup.title("Rot Win")
        tk.Label(popup, text="Angle:").pack(padx=8)
        e = tk.Entry(popup)
        e.pack(padx=8)
        e.insert(0, str(self.vp.angle))
        tk.Button(
            popup,
            text="OK",
            command=lambda: (
                self.vp.set_window_angle(float(e.get())),
                self.redraw(),
                popup.destroy(),
            ),
        ).pack(pady=8)

    def load_obj(self):
        fp = filedialog.askopenfilename(
            title="Open OBJ", filetypes=(("OBJ", "*.obj"), ("All", "*.*"))
        )
        if not fp:
            return
        v, f_idxs, scale = [], [], 50
        try:
            with open(fp, "r") as f:
                [
                    (
                        lambda p: (
                            v.append(
                                Ponto3D(
                                    float(p[1]) * scale,
                                    float(p[2]) * scale,
                                    float(p[3]) * scale,
                                )
                            )
                            if p and p[0] == "v"
                            else (
                                f_idxs.append([int(i.split("/")[0]) - 1 for i in p[1:]])
                                if p and p[0] == "f"
                                else None
                            )
                        )
                    )(line.strip().split())
                    for line in f
                ]
            if (
                not f_idxs
                and len(v) > 0
                and len(v) % 16 == 0
                and messagebox.askyesno(
                    "Bezier?",
                    f"{len(v)} verts, no faces. Load as {len(v)//16} Bezier surface(s)?",
                )
            ):
                [
                    (
                        lambda i: (
                            n := f"bezier_{fp.split('/')[-1]}_{i}",
                            o := SuperficieBezier(
                                n, v[i * 16 : (i + 1) * 16], self.sel_color
                            ),
                            self.df.adicionar(o),
                            self.obj_list.insert(tk.END, f"{n} ({o.tipo})"),
                        )
                    )(i)
                    for i in range(len(v) // 16)
                ]
                self.redraw()
                return  # CORREÇÃO: obj.tipo
            edges = set()
            [
                [
                    edges.add(tuple(sorted((face[i], face[(i + 1) % len(face)]))))
                    for i in range(len(face))
                ]
                for face in f_idxs
            ]
            n = f"obj_{fp.split('/')[-1]}"
            o = Objeto3D(n, v, list(edges), self.sel_color)
            self.df.adicionar(o)
            self.obj_list.insert(tk.END, f"{n} ({o.tipo})")
            self.redraw()  # CORREÇÃO: obj.tipo
        except Exception as e:
            messagebox.showerror("Error OBJ", f"Error: {e}")

    def choose_color(self):
        cor = colorchooser.askcolor(title="Color")[1]
        self.sel_color = cor if cor else self.sel_color
        self.cor_preview.config(bg=self.sel_color)

    def popup_add(self):  # Função adaptada para incluir SuperficieBSpline
        popup = tk.Toplevel(self)
        popup.title("Add Obj")
        popup.geometry("450x550")
        f_cor = tk.Frame(popup, pady=5)
        f_cor.pack(fill="x", padx=10)
        tk.Label(f_cor, text="Color:").pack(side="left")
        self.cor_preview = tk.Canvas(
            f_cor, width=20, height=20, bg=self.sel_color, relief="solid"
        )
        self.cor_preview.pack(side="left", padx=5)
        tk.Button(f_cor, text="Choose", command=self.choose_color).pack(side="left")
        nb = ttk.Notebook(popup)
        nb.pack(expand=True, fill="both", padx=10, pady=5)
        # --- Abas 2D ---
        f_p = ttk.Frame(nb)
        nb.add(f_p, text="Pt")
        n_p, x_p, y_p = tk.Entry(f_p), tk.Entry(f_p), tk.Entry(f_p)
        tk.Label(f_p, text="N:").grid(row=0, column=0)
        n_p.grid(row=0, column=1)
        tk.Label(f_p, text="X:").grid(row=1, column=0)
        x_p.grid(row=1, column=1)
        tk.Label(f_p, text="Y:").grid(row=2, column=0)
        y_p.grid(row=2, column=1)
        tk.Button(
            f_p,
            text="Add",
            command=lambda: self.add_obj(
                n_p.get(),
                TipoObjeto.PONTO.value,
                [(float(x_p.get()), float(y_p.get()))],
                popup,
                self.sel_color,
            ),
        ).grid(row=3, columnspan=2)
        f_r = ttk.Frame(nb)
        nb.add(f_r, text="Line")
        n_r, x1, y1, x2, y2 = (
            tk.Entry(f_r),
            tk.Entry(f_r),
            tk.Entry(f_r),
            tk.Entry(f_r),
            tk.Entry(f_r),
        )
        tk.Label(f_r, text="N:").grid(row=0, column=0)
        n_r.grid(row=0, column=1)
        tk.Label(f_r, text="X1:").grid(row=1, column=0)
        x1.grid(row=1, column=1)
        tk.Label(f_r, text="Y1:").grid(row=2, column=0)
        y1.grid(row=2, column=1)
        tk.Label(f_r, text="X2:").grid(row=3, column=0)
        x2.grid(row=3, column=1)
        tk.Label(f_r, text="Y2:").grid(row=4, column=0)
        y2.grid(row=4, column=1)
        tk.Button(
            f_r,
            text="Add",
            command=lambda: self.add_obj(
                n_r.get(),
                TipoObjeto.RETA.value,
                [
                    (float(x1.get()), float(y1.get())),
                    (float(x2.get()), float(y2.get())),
                ],
                popup,
                self.sel_color,
            ),
        ).grid(row=5, columnspan=2)
        f_poly = ttk.Frame(nb)
        nb.add(f_poly, text="Poly")
        n_poly, coords_poly = tk.Entry(f_poly), tk.Entry(f_poly, width=30)
        fill_var = tk.BooleanVar()
        tk.Label(f_poly, text="N:").grid(row=0, column=0)
        n_poly.grid(row=0, column=1)
        tk.Label(f_poly, text="Coords: (x,y),(x,y)...").grid(row=1, columnspan=2)
        coords_poly.grid(row=2, columnspan=2)
        tk.Checkbutton(f_poly, text="Fill", var=fill_var).grid(row=3, columnspan=2)
        tk.Button(
            f_poly,
            text="Add",
            command=lambda: self.add_obj(
                n_poly.get(),
                TipoObjeto.POLIGONO.value,
                list(ast.literal_eval(coords_poly.get())),
                popup,
                self.sel_color,
                fill_var.get(),
            ),
        ).grid(row=4, columnspan=2)
        f_c = ttk.Frame(nb)
        nb.add(f_c, text="Curve")
        n_c, coords_c = tk.Entry(f_c), tk.Entry(f_c, width=30)
        tk.Label(f_c, text="N:").grid(row=0, column=0)
        n_c.grid(row=0, column=1)
        tk.Label(f_c, text="Pts: (x,y),(x,y)... (4,7,10...)").grid(row=1, columnspan=2)
        coords_c.grid(row=2, columnspan=2)
        tk.Button(
            f_c,
            text="Add",
            command=lambda: self.add_obj(
                n_c.get(),
                TipoObjeto.CURVA.value,
                list(ast.literal_eval(coords_c.get())),
                popup,
                self.sel_color,
            ),
        ).grid(row=3, columnspan=2)
        f_bsp = ttk.Frame(nb)
        nb.add(f_bsp, text="BSpline")
        n_bsp, coords_bsp = tk.Entry(f_bsp), tk.Entry(f_bsp, width=30)
        tk.Label(f_bsp, text="N:").grid(row=0, column=0)
        n_bsp.grid(row=0, column=1)
        tk.Label(f_bsp, text="Pts: (x,y),(x,y)... (min 4)").grid(row=1, columnspan=2)
        coords_bsp.grid(row=2, columnspan=2)
        tk.Button(
            f_bsp,
            text="Add",
            command=lambda: self.add_obj(
                n_bsp.get(),
                TipoObjeto.B_SPLINE.value,
                list(ast.literal_eval(coords_bsp.get())),
                popup,
                self.sel_color,
            ),
        ).grid(row=3, columnspan=2)
        # --- Abas 3D ---
        f_obj3d = ttk.Frame(nb)
        nb.add(f_obj3d, text="Obj3D")
        n_obj3d = tk.Entry(f_obj3d)
        tk.Label(f_obj3d, text="N:").grid(row=0, column=0)
        n_obj3d.grid(row=0, column=1)
        tk.Label(f_obj3d, text="Verts (x,y,z);(x,y,z)...").grid(row=1, columnspan=2)
        v_txt = tk.Text(f_obj3d, height=4)
        v_txt.grid(row=2, columnspan=2)
        tk.Label(f_obj3d, text="Edges (i,j);(i,j)... base 0").grid(row=3, columnspan=2)
        e_txt = tk.Text(f_obj3d, height=4)
        e_txt.grid(row=4, columnspan=2)
        v_txt.insert(
            "1.0",
            "(-50,-50,-50);(50,-50,-50);(50,50,-50);(-50,50,-50);(-50,-50,50);(50,-50,50);(50,50,50);(-50,50,50)",
        )
        e_txt.insert(
            "1.0",
            "(0,1);(1,2);(2,3);(3,0);(4,5);(5,6);(6,7);(7,4);(0,4);(1,5);(2,6);(3,7)",
        )
        tk.Button(
            f_obj3d,
            text="Add",
            command=lambda: self.add_obj_3d(
                n_obj3d.get(),
                v_txt.get("1.0", "end-1c"),
                e_txt.get("1.0", "end-1c"),
                popup,
                self.sel_color,
            ),
        ).grid(row=5, columnspan=2)
        f_sb = ttk.Frame(nb)
        nb.add(f_sb, text="Surf Bezier")
        n_sb = tk.Entry(f_sb)
        tk.Label(f_sb, text="N:").grid(row=0, column=0)
        n_sb.grid(row=0, column=1)
        tk.Label(f_sb, text="16 Pts (x,y,z),..;(r2)..;(r3)..;(r4)").grid(
            row=1, columnspan=2
        )
        pts_sb_txt = tk.Text(f_sb, height=8)
        pts_sb_txt.grid(row=2, columnspan=2)
        tk.Button(
            f_sb,
            text="Add",
            command=lambda: self.add_superficie(
                n_sb.get(), pts_sb_txt.get("1.0", "end-1c"), popup, self.sel_color
            ),
        ).grid(row=3, columnspan=2)
        # --- Aba Superfície B-Spline (Nova) ---
        f_sbs = ttk.Frame(nb)
        nb.add(f_sbs, text="Surf BSpline")
        n_sbs = tk.Entry(f_sbs)
        tk.Label(f_sbs, text="N:").grid(row=0, column=0, sticky="w")
        n_sbs.grid(row=0, column=1, sticky="ew")
        tk.Label(f_sbs, text="Grid Pts (NxM, 4-20): (x,y,z),..;(r2).. ;..").grid(
            row=1, columnspan=2, sticky="w"
        )
        pts_sbs = tk.Text(f_sbs, height=10)
        pts_sbs.grid(row=2, columnspan=2, sticky="ew")
        tk.Label(f_sbs, text="Ns:").grid(row=3, column=0, sticky="w")
        ns_e = tk.Entry(f_sbs, width=5)
        ns_e.grid(row=3, column=1, sticky="w")
        ns_e.insert(0, "10")
        tk.Label(f_sbs, text="Nt:").grid(row=4, column=0, sticky="w")
        nt_e = tk.Entry(f_sbs, width=5)
        nt_e.grid(row=4, column=1, sticky="w")
        nt_e.insert(0, "10")
        tk.Button(
            f_sbs,
            text="Add",
            command=lambda: self.add_superficie_bspline(
                n_sbs.get(),
                pts_sbs.get("1.0", "end-1c"),
                ns_e.get(),
                nt_e.get(),
                popup,
                self.sel_color,
            ),
        ).grid(row=5, columnspan=2, pady=10)

    def add_obj(self, n, t, c, popup, cor, fill=False):
        try:
            if t == TipoObjeto.CURVA.value and (len(c) < 4 or (len(c) - 4) % 3 != 0):
                raise ValueError("Curve: 4,7,10... pts")
            if t == TipoObjeto.B_SPLINE.value and len(c) < 4:
                raise ValueError("BSpline: min 4 pts")
            coords = [(float(x), float(y)) for x, y in c]
            o = Objeto(n if n else f"{t}_{len(self.df.objetos)}", t, coords, cor, fill)
            self.df.adicionar(o)
            self.redraw()
            self.obj_list.insert(tk.END, f"{o.nome} ({o.tipo})")
            # CORREÇÃO APLICADA AQUI
            popup.destroy()
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid coords.\n\nError: {e}")

    def add_obj_3d(self, n, v_str, e_str, popup, cor):
        try:
            verts = [
                Ponto3D(*ast.literal_eval(v.strip()))
                for v in v_str.split(";")
                if v.strip()
            ]
            edges = [ast.literal_eval(e.strip()) for e in e_str.split(";") if e.strip()]
            if not verts or not edges:
                raise ValueError("Empty verts/edges")
            o = Objeto3D(n if n else f"obj3d_{len(self.df.objetos)}", verts, edges, cor)
            self.df.adicionar(o)
            self.redraw()
            self.obj_list.insert(tk.END, f"{o.nome} ({o.tipo})")
            # CORREÇÃO APLICADA AQUI
            popup.destroy()
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid 3D data.\n\nError: {e}")

    def add_superficie(self, n, pts_str, popup, cor):  # Bézier
        try:
            rows = [ln.strip() for ln in pts_str.split(";") if ln.strip()]
            if len(rows) != 4:
                raise ValueError("Need 4 rows for Bezier surface.")
            pts_raw = []
            [pts_raw.extend(ast.literal_eval(f"[{ln}]")) for ln in rows]
            if len(pts_raw) != 16:
                raise ValueError("Need 16 pts total for Bezier surface.")
            pts = [Ponto3D(x, y, z) for x, y, z in pts_raw]
            o = SuperficieBezier(n if n else f"sb_{len(self.df.objetos)}", pts, cor)
            self.df.adicionar(o)
            self.redraw()
            self.obj_list.insert(tk.END, f"{o.nome} ({o.tipo})")
            # CORREÇÃO APLICADA AQUI
            popup.destroy()
        except Exception as e:
            messagebox.showerror(
                "Input Error", f"Invalid Bezier surface data.\n\nError: {e}"
            )

    def add_superficie_bspline(self, n, grid_str, ns_str, nt_str, popup, cor):
        try:
            ns, nt = int(ns_str), int(nt_str)
            if ns < 2 or nt < 2:
                raise ValueError("Ns/Nt must be >= 2.")
            rows_str = [ls.strip() for ls in grid_str.split(";") if ls.strip()]
            rows = len(rows_str)
            if not (4 <= rows <= 20):
                raise ValueError(f"Rows ({rows}) must be 4-20.")
            grid_pts = []
            cols = -1
            for r_str in rows_str:
                pts_row_raw = ast.literal_eval(f"[{r_str}]")
                if cols == -1:
                    cols = len(pts_row_raw)
                if not (4 <= cols <= 20):
                    raise ValueError(f"Cols ({cols}) must be 4-20.")
                elif len(pts_row_raw) != cols:
                    raise ValueError("Inconsistent columns.")
                grid_pts.append([Ponto3D(x, y, z) for x, y, z in pts_row_raw])
            o = SuperficieBSpline(
                n if n else f"sbs_{len(self.df.objetos)}", grid_pts, ns, nt, cor
            )
            self.df.adicionar(o)
            self.redraw()
            self.obj_list.insert(tk.END, f"{o.nome} ({o.tipo})")
            # CORREÇÃO APLICADA AQUI (tipo já é string)
            popup.destroy()
        except Exception as e:
            messagebox.showerror(
                "Input Error BSpline Surface",
                f"Invalid data.\nCheck format, dims(4-20), Ns/Nt(>=2).\n\nError:{e}",
            )

    def popup_tf(self):  # Função unificada para abrir popups de transformação
        if not self.sel_obj_name:
            messagebox.showerror("Error", "Select object.")
            return
        o = self.df.get_by_name(self.sel_obj_name)
        if not o:
            return
        # ADIÇÃO: Inclui SuperficieBSpline no check
        if isinstance(o, (Objeto3D, SuperficieBezier, SuperficieBSpline)):
            self.popup_tf_3d(o)
        else:
            self.popup_tf_2d()

    def popup_tf_3d(self, obj):
        popup = tk.Toplevel(self)
        popup.title(f"TF 3D {obj.nome}")
        nb = ttk.Notebook(popup)
        nb.pack(expand=True, fill="both", padx=10, pady=10)
        # --- Abas de Transformação 3D ---
        ft = ttk.Frame(nb)
        nb.add(ft, text="Translate")
        dx, dy, dz = tk.Entry(ft), tk.Entry(ft), tk.Entry(ft)
        tk.Label(ft, text="dx:").grid(row=0, column=0)
        dx.grid(row=0, column=1)
        tk.Label(ft, text="dy:").grid(row=1, column=0)
        dy.grid(row=1, column=1)
        tk.Label(ft, text="dz:").grid(row=2, column=0)
        dz.grid(row=2, column=1)
        tk.Button(
            ft,
            text="Apply",
            command=lambda: self.apply_tf_3d(
                obj,
                "translacao",
                dx=float(dx.get() or 0),
                dy=float(dy.get() or 0),
                dz=float(dz.get() or 0),
                popup=popup,
            ),
        ).grid(row=3, columnspan=2)
        fs = ttk.Frame(nb)
        nb.add(fs, text="Scale")
        sx, sy, sz = tk.Entry(fs), tk.Entry(fs), tk.Entry(fs)
        tk.Label(fs, text="sx:").grid(row=0, column=0)
        sx.grid(row=0, column=1)
        tk.Label(fs, text="sy:").grid(row=1, column=0)
        sy.grid(row=1, column=1)
        tk.Label(fs, text="sz:").grid(row=2, column=0)
        sz.grid(row=2, column=1)
        tk.Button(
            fs,
            text="Apply",
            command=lambda: self.apply_tf_3d(
                obj,
                "escalonamento",
                sx=float(sx.get() or 1),
                sy=float(sy.get() or 1),
                sz=float(sz.get() or 1),
                popup=popup,
            ),
        ).grid(row=3, columnspan=2)
        fr = ttk.Frame(nb)
        nb.add(fr, text="Rotate")
        angle = tk.Entry(fr)
        axis = tk.StringVar(value="y")
        tk.Label(fr, text="Angle:").grid(row=0, column=0)
        angle.grid(row=0, column=1)
        tk.Label(fr, text="Axis:").grid(row=1, column=0)
        tk.Radiobutton(fr, text="X", var=axis, value="x").grid(
            row=1, column=1, sticky="w"
        )
        tk.Radiobutton(fr, text="Y", var=axis, value="y").grid(
            row=2, column=1, sticky="w"
        )
        tk.Radiobutton(fr, text="Z", var=axis, value="z").grid(
            row=3, column=1, sticky="w"
        )
        tk.Button(
            fr,
            text="Apply",
            command=lambda: self.apply_tf_3d(
                obj,
                "rotacao",
                angulo=float(angle.get() or 0),
                eixo=axis.get(),
                popup=popup,
            ),
        ).grid(row=4, columnspan=2)

    def popup_tf_2d(self):
        self.obj_list.unbind("<<ListboxSelect>>")
        popup = tk.Toplevel(self)
        popup.title(f"TF 2D {self.sel_obj_name}")
        popup.geometry("350x300")
        popup.protocol("WM_DELETE_WINDOW", lambda: self.close_tf_popup(popup))
        nb = ttk.Notebook(popup)
        nb.pack(expand=True, fill="both", padx=10, pady=10)
        # --- Abas de Transformação 2D ---
        ft = ttk.Frame(nb)
        nb.add(ft, text="Translate")
        dx, dy = tk.Entry(ft), tk.Entry(ft)
        tk.Label(ft, text="dx:").grid(row=0, column=0)
        dx.grid(row=0, column=1)
        tk.Label(ft, text="dy:").grid(row=1, column=0)
        dy.grid(row=1, column=1)
        tk.Button(
            ft,
            text="Apply",
            command=lambda: self.apply_tf_2d(
                "translacao",
                dx=float(dx.get() or 0),
                dy=float(dy.get() or 0),
                popup=popup,
            ),
        ).grid(row=2, columnspan=2)
        fs = ttk.Frame(nb)
        nb.add(fs, text="Scale")
        sx, sy = tk.Entry(fs), tk.Entry(fs)
        tk.Label(fs, text="sx:").grid(row=0, column=0)
        sx.grid(row=0, column=1)
        tk.Label(fs, text="sy:").grid(row=1, column=0)
        sy.grid(row=1, column=1)
        tk.Button(
            fs,
            text="Apply",
            command=lambda: self.apply_tf_2d(
                "escalonamento_natural",
                sx=float(sx.get() or 1),
                sy=float(sy.get() or 1),
                popup=popup,
            ),
        ).grid(row=2, columnspan=2)
        fr = ttk.Frame(nb)
        nb.add(fr, text="Rotate")
        tk.Label(fr, text="Angle:").grid(row=0, column=0)
        angle = tk.Entry(fr)
        angle.grid(row=0, column=1)
        rot_var = tk.StringVar(value="centro_objeto")
        tk.Radiobutton(fr, text="Center", var=rot_var, value="centro_objeto").grid(
            row=1, sticky="w"
        )
        tk.Radiobutton(fr, text="Origin", var=rot_var, value="centro_mundo").grid(
            row=2, sticky="w"
        )
        f_arb = tk.Frame(fr)
        f_arb.grid(row=3, sticky="w")
        tk.Radiobutton(f_arb, text="Pt:", var=rot_var, value="ponto_arbitrario").pack(
            side="left"
        )
        xr, yr = tk.Entry(f_arb, width=5), tk.Entry(f_arb, width=5)
        tk.Label(f_arb, text="x:").pack(side="left")
        xr.pack(side="left")
        tk.Label(f_arb, text="y:").pack(side="left")
        yr.pack(side="left")
        tk.Button(
            fr,
            text="Apply",
            command=lambda: self.apply_tf_2d(
                rot_var.get(),
                angulo=float(angle.get() or 0),
                px=float(xr.get() or 0),
                py=float(yr.get() or 0),
                popup=popup,
            ),
        ).grid(row=4, columnspan=2)

    def close_tf_popup(self, p):
        self.obj_list.bind("<<ListboxSelect>>", self.on_sel)
        p.destroy()

    def apply_tf_3d(self, obj, tipo, **kwargs):
        popup = kwargs.pop("popup", None)
        try:
            # ADIÇÃO: Lógica para calcular centro de SuperficieBSpline
            cx, cy, cz = 0, 0, 0
            if tipo in ["escalonamento", "rotacao"]:
                if isinstance(obj, Objeto3D):
                    cx, cy, cz = obj.get_centro_objeto_3d()
                elif isinstance(obj, SuperficieBezier):
                    cx, cy, cz = np.mean(
                        [p.coords[:3] for p in obj.pontos_controle.flatten()], axis=0
                    )
                elif isinstance(obj, SuperficieBSpline):
                    cx, cy, cz = np.mean(
                        [p.coords[:3] for r in obj.grid for p in r], axis=0
                    )  # Média dos pontos de controle

            if tipo == "translacao":
                m = Transformacoes3D.get_matriz_translacao(
                    kwargs["dx"], kwargs["dy"], kwargs["dz"]
                )
            elif tipo == "escalonamento":
                m = (
                    Transformacoes3D.get_matriz_translacao(cx, cy, cz)
                    @ Transformacoes3D.get_matriz_escalonamento(
                        kwargs["sx"], kwargs["sy"], kwargs["sz"]
                    )
                    @ Transformacoes3D.get_matriz_translacao(-cx, -cy, -cz)
                )
            elif tipo == "rotacao":
                a = np.deg2rad(kwargs["angulo"])
                e = kwargs["eixo"]
                r = (
                    Transformacoes3D.get_matriz_rotacao_x(a)
                    if e == "x"
                    else (
                        Transformacoes3D.get_matriz_rotacao_y(a)
                        if e == "y"
                        else Transformacoes3D.get_matriz_rotacao_z(a)
                    )
                )
                m = (
                    Transformacoes3D.get_matriz_translacao(cx, cy, cz)
                    @ r
                    @ Transformacoes3D.get_matriz_translacao(-cx, -cy, -cz)
                )

            obj.aplicar_transformacao_3d(m)
            self.redraw()
            if popup:
                popup.destroy()
        except Exception as e:
            messagebox.showerror("Error TF3D", f"Invalid input: {e}")

    def apply_tf_2d(self, tipo, **kwargs):
        o = self.df.get_by_name(self.sel_obj_name)
        popup = kwargs.pop("popup", None)
        if not o:
            return
        try:
            if tipo == "translacao":
                self.tf2d.aplicar_transformacao_generica(
                    o, self.tf2d.get_matriz_translacao(kwargs["dx"], kwargs["dy"])
                )
            elif tipo == "escalonamento_natural":
                self.tf2d.aplicar_escalonamento_natural(o, kwargs["sx"], kwargs["sy"])
            elif tipo == "centro_mundo":
                self.tf2d.aplicar_transformacao_generica(
                    o, self.tf2d.get_matriz_rotacao(np.deg2rad(kwargs["angulo"]))
                )
            elif tipo == "centro_objeto":
                self.tf2d.aplicar_rotacao_centro_objeto(o, np.deg2rad(kwargs["angulo"]))
            elif tipo == "ponto_arbitrario":
                self.tf2d.aplicar_rotacao_ponto_arbitrario(
                    o, np.deg2rad(kwargs["angulo"]), kwargs["px"], kwargs["py"]
                )
            self.redraw()
            if popup:
                self.close_tf_popup(popup)
        except Exception as e:
            messagebox.showerror("Error TF2D", f"Invalid input: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("SGI 2D/3D - Proj & Superfícies")
    root.geometry("1200x800")
    app = App(root)
    app.mainloop()
