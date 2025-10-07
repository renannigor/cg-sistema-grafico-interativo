import tkinter as tk
from tkinter import ttk, colorchooser, messagebox, simpledialog
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
        return np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def get_matriz_escalonamento(sx, sy, sz):
        return np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def get_matriz_rotacao_x(angulo_rad):
        c = np.cos(angulo_rad)
        s = np.sin(angulo_rad)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def get_matriz_rotacao_y(angulo_rad):
        c = np.cos(angulo_rad)
        s = np.sin(angulo_rad)
        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def get_matriz_rotacao_z(angulo_rad):
        c = np.cos(angulo_rad)
        s = np.sin(angulo_rad)
        return np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

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
            Rx = np.array([
                [1, 0, 0, 0],
                [0, c/d, -b/d, 0],
                [0, b/d, c/d, 0],
                [0, 0, 0, 1]
            ])
        Rx_inv = Rx.T
        
        Ry = np.array([
            [d, 0, -a, 0],
            [0, 1, 0, 0],
            [a, 0, d, 0],
            [0, 0, 0, 1]
        ])
        Ry_inv = Ry.T
        
        Rz = cls.get_matriz_rotacao_z(angulo_rad)
        
        return T_inv @ Rx_inv @ Ry_inv @ Rz @ Ry @ Rx @ T

class Camera:
    """Gerencia a visualização 3D (VRP, VPN, VUP)."""
    def __init__(self, vrp: Ponto3D, p_foco: Ponto3D, vup: tuple):
        self.vrp = vrp
        self.p_foco = p_foco
        self.vup = np.array(vup)
        self.matriz_wc = self._calcular_matriz_wc()

    def _calcular_matriz_wc(self):
        """Calcula a matriz de transformação Mundo -> Câmera."""
        vpn = self.p_foco.coords[:3] - self.vrp.coords[:3]
        
        n = Transformacoes3D._normalizar(vpn)
        u = Transformacoes3D._normalizar(np.cross(self.vup, n))
        v = Transformacoes3D._normalizar(np.cross(n, u))
        
        matriz_rot = np.array([
            np.append(u, 0),
            np.append(v, 0),
            np.append(n, 0),
            [0, 0, 0, 1]
        ])
        
        matriz_trans = Transformacoes3D.get_matriz_translacao(-self.vrp.x, -self.vrp.y, -self.vrp.z)
        
        return matriz_rot @ matriz_trans

    def get_matriz_wc(self):
        return self.matriz_wc

    def mover(self, dx, dy, dz):
        """Move VRP e Ponto de Foco juntos."""
        self.vrp.aplicar_transformacao(Transformacoes3D.get_matriz_translacao(dx, dy, dz))
        self.p_foco.aplicar_transformacao(Transformacoes3D.get_matriz_translacao(dx, dy, dz))
        self.matriz_wc = self._calcular_matriz_wc()

    def girar(self, angulo_y_rad, angulo_x_rad):
        """Gira a câmera em torno do VRP (Yaw e Pitch)."""
        vpn = self.p_foco.coords[:3] - self.vrp.coords[:3]
        n = Transformacoes3D._normalizar(vpn)
        u = Transformacoes3D._normalizar(np.cross(self.vup, n))

        rot_y = Transformacoes3D.get_matriz_rotacao_y(angulo_y_rad)

        rot_x_matriz = Transformacoes3D.get_matriz_rotacao_eixo_arbitrario(
            (0,0,0), u, angulo_x_rad
        )
        
        novo_p_foco_coords = rot_y @ rot_x_matriz @ np.append(vpn, 1)
        
        self.p_foco = Ponto3D(
            self.vrp.x + novo_p_foco_coords[0],
            self.vrp.y + novo_p_foco_coords[1],
            self.vrp.z + novo_p_foco_coords[2]
        )
        self.matriz_wc = self._calcular_matriz_wc()



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


class TipoObjeto(Enum):
    PONTO = "ponto"
    RETA = "reta"
    POLIGONO = "poligono"
    CURVA = "curva"
    B_SPLINE = "b-spline"
    OBJETO_3D = "objeto_3d"

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
        self.coords = coords # Para objetos 2D
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


class DisplayFile:
    def __init__(self):
        self.objetos = []

    def adicionar(self, obj):
        self.objetos.append(obj)

    def listar(self):
        return self.objetos

    def get_by_name(self, nome):
        for obj in self.objetos:
            if obj.nome == nome:
                return obj
        return None

class GeradorBSpline:
    M_BSPLINE = (1/6) * np.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  0,  3, 0],
        [ 1,  4,  1, 0]
    ])

    @staticmethod
    def _get_matriz_E(delta: float) -> np.ndarray:
        d2 = delta * delta
        d3 = d2 * delta
        return np.array([
            [0,    0,    0,     1],
            [d3,   d2,   delta, 0],
            [6*d3, 2*d2, 0,     0],
            [6*d3, 0,    0,     0]
        ])

    def gerar_segmentos(self, pontos_controle: List[Tuple[float, float]], delta: float = 0.1) -> List[List[Tuple[float, float]]]:
        if len(pontos_controle) < 4:
            return []

        segmentos_de_reta = []
        num_curvas = len(pontos_controle) - 3
        matriz_E = self._get_matriz_E(delta)
        num_passos = int(1 / delta)

        for i in range(num_curvas):
            p_controle_segmento = pontos_controle[i : i + 4]
            
            Gx = np.array([p[0] for p in p_controle_segmento]).reshape(4, 1)
            Gy = np.array([p[1] for p in p_controle_segmento]).reshape(4, 1)

            Cx = self.M_BSPLINE @ Gx
            Cy = self.M_BSPLINE @ Gy
            
            Dx = (matriz_E @ Cx).flatten()
            Dy = (matriz_E @ Cy).flatten()

            x, dx, d2x, d3x = Dx[0], Dx[1], Dx[2], Dx[3]
            y, dy, d2y, d3y = Dy[0], Dy[1], Dy[2], Dy[3]

            for _ in range(num_passos):
                x_velho, y_velho = x, y
                
                x += dx
                dx += d2x
                d2x += d3x
                y += dy
                dy += d2y
                d2y += d3y

                segmentos_de_reta.append([(x_velho, y_velho), (x, y)])

        return segmentos_de_reta

class Clipping:
    INSIDE = 0; LEFT = 1; RIGHT = 2; BOTTOM = 4; TOP = 8

    def _get_outcode(self, x, y, wmin, wmax):
        code = self.INSIDE
        if x < wmin[0]: code |= self.LEFT
        elif x > wmax[0]: code |= self.RIGHT
        if y < wmin[1]: code |= self.BOTTOM
        elif y > wmax[1]: code |= self.TOP
        return code

    def cohen_sutherland(self, p1, p2, wmin, wmax):
        x1, y1 = p1
        x2, y2 = p2
        outcode1 = self._get_outcode(x1, y1, wmin, wmax)
        outcode2 = self._get_outcode(x2, y2, wmin, wmax)
        
        while True:
            if not (outcode1 | outcode2): return [(x1, y1), (x2, y2)]
            elif outcode1 & outcode2: return []
            else:
                x, y = 0, 0
                outcode_fora = outcode1 if outcode1 else outcode2
                if outcode_fora & self.TOP:
                    x = x1 + (x2 - x1) * (wmax[1] - y1) / (y2 - y1)
                    y = wmax[1]
                elif outcode_fora & self.BOTTOM:
                    x = x1 + (x2 - x1) * (wmin[1] - y1) / (y2 - y1)
                    y = wmin[1]
                elif outcode_fora & self.RIGHT:
                    y = y1 + (y2 - y1) * (wmax[0] - x1) / (x2 - x1)
                    x = wmax[0]
                elif outcode_fora & self.LEFT:
                    y = y1 + (y2 - y1) * (wmin[0] - x1) / (x2 - x1)
                    x = wmin[0]

                if outcode_fora == outcode1:
                    x1, y1 = x, y
                    outcode1 = self._get_outcode(x1, y1, wmin, wmax)
                else:
                    x2, y2 = x, y
                    outcode2 = self._get_outcode(x2, y2, wmin, wmax)

    def liang_barsky(self, p1, p2, wmin, wmax):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        p = [-dx, dx, -dy, dy]
        q = [x1 - wmin[0], wmax[0] - x1, y1 - wmin[1], wmax[1] - y1]
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
        
        if u1 > u2: return []

        clipped_p1 = (x1 + u1 * dx, y1 + u1 * dy)
        clipped_p2 = (x1 + u2 * dx, y1 + u2 * dy)
        return [clipped_p1, clipped_p2]

    def sutherland_hodgman(self, vertices, wmin, wmax):
        def clip_contra_aresta(verts, aresta):
            novos_verts = []
            for i in range(len(verts)):
                p1, p2 = verts[i], verts[(i + 1) % len(verts)]
                inside1, inside2 = esta_dentro(p1, aresta), esta_dentro(p2, aresta)
                if inside1 and inside2: novos_verts.append(p2)
                elif inside1 and not inside2: novos_verts.append(intersecao(p1, p2, aresta))
                elif not inside1 and inside2:
                    novos_verts.append(intersecao(p1, p2, aresta))
                    novos_verts.append(p2)
            return novos_verts

        def esta_dentro(p, aresta):
            if aresta == 'left': return p[0] >= wmin[0]
            if aresta == 'right': return p[0] <= wmax[0]
            if aresta == 'bottom': return p[1] >= wmin[1]
            if aresta == 'top': return p[1] <= wmax[1]
        
         def intersecao(p1, p2, aresta):
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1
            if aresta == 'left':
                y = y1 + dy * (wmin[0] - x1) / dx if dx != 0 else y1
                return (wmin[0], y)
            if aresta == 'right':
                y = y1 + dy * (wmax[0] - x1) / dx if dx != 0 else y1
                return (wmax[0], y)
            if aresta == 'bottom':
                x = x1 + dx * (wmin[1] - y1) / dy if dy != 0 else x1
                return (x, wmin[1])
            if aresta == 'top':
                x = x1 + dx * (wmax[1] - y1) / dy if dy != 0 else x1
                return (x, wmax[1])
        
        clipped_vertices = vertices
        for aresta in ['left', 'right', 'bottom', 'top']:
            clipped_vertices = clip_contra_aresta(clipped_vertices, aresta)
        return clipped_vertices

    def _gerar_pontos_bezier(self, pontos_controle, num_passos=20):
        pontos = []
        p1, p2, p3, p4 = [np.array(p) for p in pontos_controle]
        for i in range(num_passos + 1):
            t = i / num_passos
            ponto = (1-t)**3 * p1 + 3*(1-t)**2 * t * p2 + 3*(1-t) * t**2 * p3 + t**3 * p4
            pontos.append(tuple(ponto))
        return pontos

    def _subdivide_bezier(self, p, t):
        p1, p2, p3, p4 = p
        p12 = (1 - t) * np.array(p1) + t * np.array(p2)
        p23 = (1 - t) * np.array(p2) + t * np.array(p3)
        p34 = (1 - t) * np.array(p3) + t * np.array(p4)
        p123 = (1 - t) * p12 + t * p23
        p234 = (1 - t) * p23 + t * p34
        p1234 = (1 - t) * p123 + t * p234
        
        curva1 = [p1, tuple(p12), tuple(p123), tuple(p1234)]
        curva2 = [tuple(p1234), tuple(p234), tuple(p34), p4]
        return curva1, curva2

    def _clip_bezier_recursivo(self, pontos_controle, wmin, wmax, profundidade=0):
        if profundidade > 10:
            linha = self.cohen_sutherland(pontos_controle[0], pontos_controle[-1], wmin, wmax)
            return [linha] if linha else []

        todos_dentro = all(self._get_outcode(p[0], p[1], wmin, wmax) == self.INSIDE for p in pontos_controle)
        if todos_dentro:
            pontos_gerados = self._gerar_pontos_bezier(pontos_controle)
            segmentos = []
            for i in range(len(pontos_gerados) - 1):
                segmentos.append([pontos_gerados[i], pontos_gerados[i+1]])
            return segmentos

        outcode_comum = 0b1111
        for p in pontos_controle:
            outcode_comum &= self._get_outcode(p[0], p[1], wmin, wmax)
        if outcode_comum != self.INSIDE:
            return []

        curva1, curva2 = self._subdivide_bezier(pontos_controle, 0.5)
        
        segmentos_clipados = []
        segmentos_clipados.extend(self._clip_bezier_recursivo(curva1, wmin, wmax, profundidade + 1))
        segmentos_clipados.extend(self._clip_bezier_recursivo(curva2, wmin, wmax, profundidade + 1))
        return segmentos_clipados

    def clip_curva_para_linhas(self, pontos_controle_total, wmin, wmax):
        segmentos_de_linha_visiveis = []
        if len(pontos_controle_total) < 4: return []
        
        num_segmentos = (len(pontos_controle_total) - 4) // 3 + 1
        
        for i in range(num_segmentos):
            inicio_idx = i * 3
            pontos_segmento_atual = pontos_controle_total[inicio_idx:inicio_idx + 4]
            linhas_clipadas = self._clip_bezier_recursivo(pontos_segmento_atual, wmin, wmax)
            segmentos_de_linha_visiveis.extend(linhas_clipadas)
            
        return segmentos_de_linha_visiveis

    def clip(self, obj, wmin, wmax, alg_reta):
        coords_clipadas = []
        if obj.tipo == TipoObjeto.PONTO.value:
            x, y = obj.coords[0]
            if wmin[0] <= x <= wmax[0] and wmin[1] <= y <= wmax[1]:
                coords_clipadas = obj.coords
        
        elif obj.tipo == TipoObjeto.RETA.value:
            if alg_reta == 'cs':
                coords_clipadas = self.cohen_sutherland(obj.coords[0], obj.coords[1], wmin, wmax)
            else:
                coords_clipadas = self.liang_barsky(obj.coords[0], obj.coords[1], wmin, wmax)
        
        elif obj.tipo == TipoObjeto.POLIGONO.value:
            coords_clipadas = self.sutherland_hodgman(obj.coords, wmin, wmax)

        if not coords_clipadas: return None

        if obj.tipo in [TipoObjeto.CURVA.value, TipoObjeto.B_SPLINE.value, TipoObjeto.OBJETO_3D.value]:
            return obj

        return Objeto(obj.nome, obj.tipo, coords_clipadas, obj.cor, obj.preenchido)


class Viewport:
    def __init__(self, canvas, wmin, wmax, camera):
        self.canvas = canvas
        self.wmin = wmin
        self.wmax = wmax
        self.window_angle = 0.0
        self.clipping = Clipping()
        self.bspline_generator = GeradorBSpline()
        self.camera = camera
        
    def set_window_angle(self, ang_deg):
        self.window_angle = ang_deg

    def _rotacionar_ponto_em_torno_de(self, x, y, cx, cy, ang_rad):
        tx, ty = x - cx, y - cy
        c, s = np.cos(ang_rad), np.sin(ang_rad)
        xr, yr = c * tx - s * ty, s * tx + c * ty
        return (xr + cx, yr + cy)

    def transformar(self, xw, yw, w_min, w_max, vpmin, vpmax):
        xw_range, yw_range = w_max[0] - w_min[0], w_max[1] - w_min[1]
        xvp_range, yvp_range = vpmax[0] - vpmin[0], vpmax[1] - vpmin[1]

        if xw_range == 0 or yw_range == 0: return (0, 0)

        xvp = vpmin[0] + ((xw - w_min[0]) / xw_range) * xvp_range
        yvp = vpmin[1] + (1 - (yw - w_min[1]) / yw_range) * yvp_range
        return (xvp, yvp)

    def desenhar(self, displayfile: DisplayFile, alg_reta_clip: str):
        self.canvas.delete("all")
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        canvas_margin = 10
        if canvas_width <= (2 * canvas_margin) or canvas_height <= (2 * canvas_margin): return

        available_width, available_height = canvas_width - (2*canvas_margin), canvas_height - (2*canvas_margin)
        w_width, w_height = self.wmax[0] - self.wmin[0], self.wmax[1] - self.wmin[1]
        
        w_aspect = w_width / w_height if w_height != 0 else 1.0
        available_aspect = available_width / available_height

        if available_aspect > w_aspect:
            vp_height = available_height
            vp_width = vp_height * w_aspect
            offset_x = canvas_margin + (available_width - vp_width) / 2
            offset_y = canvas_margin
        else:
            vp_width = available_width
            vp_height = vp_width / w_aspect
            offset_x = canvas_margin
            offset_y = canvas_margin + (available_height - vp_height) / 2

        vpmin = (offset_x, offset_y)
        vpmax = (offset_x + vp_width, offset_y + vp_height)
        
        self.canvas.create_rectangle(vpmin[0], vpmin[1], vpmax[0], vpmax[1], outline="blue", dash=(4, 4))
        
        cx, cy = (self.wmin[0] + self.wmax[0])/2.0, (self.wmin[1] + self.wmax[1])/2.0
        ang_rad = -np.deg2rad(self.window_angle)

        def transform_coord(xw, yw):
            xr, yr = self._rotacionar_ponto_em_torno_de(xw, yw, cx, cy, ang_rad)
            return self.transformar(xr, yr, self.wmin, self.wmax, vpmin, vpmax)

        origem_vp = transform_coord(0, 0)
        p_inicio_eixo_x, p_fim_eixo_x = transform_coord(self.wmin[0], 0), transform_coord(self.wmax[0], 0)
        p_inicio_eixo_y, p_fim_eixo_y = transform_coord(0, self.wmin[1]), transform_coord(0, self.wmax[1])
        
        self.canvas.create_line(p_inicio_eixo_x[0], origem_vp[1], p_fim_eixo_x[0], origem_vp[1], fill="gray", dash=(2, 2))
        self.canvas.create_line(origem_vp[0], p_inicio_eixo_y[1], origem_vp[0], p_fim_eixo_y[1], fill="gray", dash=(2, 2))

        for obj in displayfile.listar():
            if obj.tipo == TipoObjeto.OBJETO_3D.value:
                obj_camera_coords = copy.deepcopy(obj)
                matriz_wc = self.camera.get_matriz_wc()
                obj_camera_coords.aplicar_transformacao_3d(matriz_wc)

                for idx1, idx2 in obj_camera_coords.arestas:
                    p1_cam = obj_camera_coords.vertices[idx1]
                    p2_cam = obj_camera_coords.vertices[idx2]

                    p1_2d, p2_2d = (p1_cam.x, p1_cam.y), (p2_cam.x, p2_cam.y)

                    reta_clipada_coords = self.clipping.cohen_sutherland(p1_2d, p2_2d, self.wmin, self.wmax)
                    if reta_clipada_coords:
                        p1_vp = transform_coord(reta_clipada_coords[0][0], reta_clipada_coords[0][1])
                        p2_vp = transform_coord(reta_clipada_coords[1][0], reta_clipada_coords[1][1])
                        self.canvas.create_line(p1_vp, p2_vp, fill=obj.cor, width=2)
                continue # Pula para o próximo objeto

            if obj.tipo == TipoObjeto.CURVA.value:
                segmentos_de_linha = self.clipping.clip_curva_para_linhas(obj.coords, self.wmin, self.wmax)
                for linha in segmentos_de_linha:
                    p1_vp, p2_vp = transform_coord(*linha[0]), transform_coord(*linha[1])
                    self.canvas.create_line(p1_vp, p2_vp, fill=obj.cor, width=2)
                continue
            
            elif obj.tipo == TipoObjeto.B_SPLINE.value:
                segmentos_de_reta = self.bspline_generator.gerar_segmentos(obj.coords)
                for p1, p2 in segmentos_de_reta:
                    reta_clipada_coords = self.clipping.cohen_sutherland(p1, p2, self.wmin, self.wmax)
                    if reta_clipada_coords:
                        p1_vp, p2_vp = transform_coord(*reta_clipada_coords[0]), transform_coord(*reta_clipada_coords[1])
                        self.canvas.create_line(p1_vp, p2_vp, fill=obj.cor, width=2)
                continue

            obj_clipado = self.clipping.clip(obj, self.wmin, self.wmax, alg_reta_clip)
            if not obj_clipado: continue

            coords_vp = [transform_coord(x, y) for (x, y) in obj_clipado.coords]

            if obj_clipado.tipo == TipoObjeto.PONTO.value:
                x, y = coords_vp[0]
                self.canvas.create_oval(x-2, y-2, x+2, y+2, fill=obj_clipado.cor, outline=obj_clipado.cor)
            elif obj_clipado.tipo == TipoObjeto.RETA.value:
                flat = [c for p in coords_vp for c in p]
                if len(flat) >= 4: self.canvas.create_line(*flat, fill=obj_clipado.cor, width=2)
            elif obj_clipado.tipo == TipoObjeto.POLIGONO.value:
                if len(coords_vp) > 1:
                    fill_color = obj_clipado.cor if obj_clipado.preenchido else ""
                    outline_color = "" if obj_clipado.preenchido else obj_clipado.cor
                    self.canvas.create_polygon(coords_vp, fill=fill_color, outline=outline_color, width=2)


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        self.selected_obj_name = None
        self.transformacoes = Transformacoes()
        self.displayfile = DisplayFile()
        
        self.camera = Camera(vrp=Ponto3D(0, 50, 200), p_foco=Ponto3D(0, 0, 0), vup=(0, 1, 0))
        
        frame_main = tk.Frame(self)
        frame_main.pack(fill="both", expand=True)
        frame_main.grid_columnconfigure(0, weight=1)
        frame_main.grid_columnconfigure(1, weight=0)
        frame_main.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame_main, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.viewport = Viewport(self.canvas, wmin=(-100, -100), wmax=(100, 100), camera=self.camera)

        self.lista_objetos = tk.Listbox(frame_main, width=30)
        self.lista_objetos.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.lista_objetos.bind("<<ListboxSelect>>", self.on_obj_select)

        main_controls_container = tk.Frame(self)
        main_controls_container.pack(fill="x", padx=5, pady=(0, 5))
        top_row = tk.Frame(main_controls_container)
        top_row.pack(fill="x")
        mid_row = tk.Frame(main_controls_container)
        mid_row.pack(fill="x", pady=5)
        bottom_row = tk.Frame(main_controls_container)
        bottom_row.pack(fill="x")

        tk.Button(top_row, text="Adicionar Objeto", command=self.abrir_popup_objetos).pack(side="left", padx=2)
        tk.Button(top_row, text="Aplicar Transformação", command=self.abrir_transformacoes_popup).pack(side="left", padx=2)
        tk.Label(top_row, text="  Nav. 2D:").pack(side="left", padx=(10,0))
        tk.Button(top_row, text="←", command=lambda: self.pan(10, 0)).pack(side="left")
        tk.Button(top_row, text="→", command=lambda: self.pan(-10, 0)).pack(side="left")
        tk.Button(top_row, text="↑", command=lambda: self.pan(0, -10)).pack(side="left")
        tk.Button(top_row, text="↓", command=lambda: self.pan(0, 10)).pack(side="left")
        tk.Button(top_row, text="Zoom +", command=lambda: self.zoom(0.9)).pack(side="left", padx=(5,2))
        tk.Button(top_row, text="Zoom -", command=lambda: self.zoom(1.1)).pack(side="left")
        tk.Button(top_row, text="Rot. Win", command=self.popup_rotacionar_window).pack(side="left", padx=(5,2))

        tk.Label(mid_row, text="Nav. 3D (Câmera):").pack(side="left")
        tk.Button(mid_row, text="Frente", command=lambda: (self.camera.mover(0,0,-10), self.redesenhar())).pack(side="left")
        tk.Button(mid_row, text="Trás", command=lambda: (self.camera.mover(0,0,10), self.redesenhar())).pack(side="left")
        tk.Button(mid_row, text="Esq.", command=lambda: (self.camera.mover(-10,0,0), self.redesenhar())).pack(side="left")
        tk.Button(mid_row, text="Dir.", command=lambda: (self.camera.mover(10,0,0), self.redesenhar())).pack(side="left")
        tk.Button(mid_row, text="Cima", command=lambda: (self.camera.mover(0,10,0), self.redesenhar())).pack(side="left")
        tk.Button(mid_row, text="Baixo", command=lambda: (self.camera.mover(0,-10,0), self.redesenhar())).pack(side="left")
        tk.Button(mid_row, text="⟲ (Yaw)", command=lambda: (self.camera.girar(np.deg2rad(5),0), self.redesenhar())).pack(side="left", padx=(5,0))
        tk.Button(mid_row, text="⟳ (Yaw)", command=lambda: (self.camera.girar(np.deg2rad(-5),0), self.redesenhar())).pack(side="left")
        tk.Button(mid_row, text="⬆ (Pitch)", command=lambda: (self.camera.girar(0, np.deg2rad(5)), self.redesenhar())).pack(side="left")
        tk.Button(mid_row, text="⬇ (Pitch)", command=lambda: (self.camera.girar(0, np.deg2rad(-5)), self.redesenhar())).pack(side="left")

        # Bottom Row - Clipping
        clipping_frame = tk.Frame(bottom_row, borderwidth=1, relief="groove")
        clipping_frame.pack()
        tk.Label(clipping_frame, text="Clipping de Reta:").pack(side="left", padx=5)
        self.alg_reta_clip_var = tk.StringVar(value="cs")
        tk.Radiobutton(clipping_frame, text="Cohen-Sutherland", variable=self.alg_reta_clip_var, value="cs", command=self.redesenhar).pack(side="left")
        tk.Radiobutton(clipping_frame, text="Liang-Barsky", variable=self.alg_reta_clip_var, value="lb", command=self.redesenhar).pack(side="left", padx=(0,5))

        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.master.bind("<KeyPress>", self.on_key_press) 
        self.selected_color = "black"

    def on_key_press(self, event):
        """Redesenha ao pressionar qualquer tecla para atualizar a câmera."""
        self.redesenhar()

    def redesenhar(self):
        self.viewport.desenhar(self.displayfile, self.alg_reta_clip_var.get())

    def popup_rotacionar_window(self):
        popup = tk.Toplevel(self)
        popup.title("Rotacionar Window")
        tk.Label(popup, text="Ângulo (graus, anti-horário):").pack(padx=8, pady=(8,0))
        ang_entry = tk.Entry(popup)
        ang_entry.pack(padx=8, pady=4)
        ang_entry.insert(0, str(self.viewport.window_angle))

        def aplicar():
            try:
                self.viewport.set_window_angle(float(ang_entry.get()))
                self.redesenhar()
                popup.destroy()
            except Exception as e:
                messagebox.showerror("Erro", f"Ângulo inválido: {e}")
        tk.Button(popup, text="Aplicar", command=aplicar).pack(pady=8)

    def on_canvas_resize(self, event): self.redesenhar()

    def on_obj_select(self, event):
        selecao = self.lista_objetos.curselection()
        if selecao:
            entry = self.lista_objetos.get(selecao[0])
            self.selected_obj_name = entry.rsplit(" (", 1)[0] if " (" in entry else entry
        else: self.selected_obj_name = None

    def abrir_popup_objetos(self):
        popup = tk.Toplevel(self)
        popup.title("Incluir Objeto")
        popup.geometry("350x400")

        frame_cor = tk.Frame(popup, pady=5)
        frame_cor.pack(fill="x", padx=10)
        tk.Label(frame_cor, text="Cor:").pack(side="left")
        self.cor_preview = tk.Canvas(frame_cor, width=20, height=20, bg="black", relief="solid")
        self.cor_preview.pack(side="left", padx=5)
        tk.Button(frame_cor, text="Escolher Cor", command=self.escolher_cor).pack(side="left")

        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=5)
        
        # Aba Ponto
        frame_ponto = ttk.Frame(notebook)
        notebook.add(frame_ponto, text="Ponto")
        nome_ponto, x_ponto, y_ponto = tk.Entry(frame_ponto), tk.Entry(frame_ponto), tk.Entry(frame_ponto)
        tk.Label(frame_ponto, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_ponto.grid(row=0, column=1)
        tk.Label(frame_ponto, text="x:").grid(row=1, column=0, sticky="w", pady=2)
        x_ponto.grid(row=1, column=1)
        tk.Label(frame_ponto, text="y:").grid(row=2, column=0, sticky="w", pady=2)
        y_ponto.grid(row=2, column=1)
        tk.Button(frame_ponto, text="Adicionar", command=lambda: self.add_obj(nome_ponto.get(), TipoObjeto.PONTO.value, [(float(x_ponto.get()), float(y_ponto.get()))], popup, self.selected_color)).grid(row=3, columnspan=2, pady=10)

        # Aba Reta
        frame_reta = ttk.Frame(notebook)
        notebook.add(frame_reta, text="Reta")
        nome_reta, x1, y1, x2, y2 = tk.Entry(frame_reta), tk.Entry(frame_reta), tk.Entry(frame_reta), tk.Entry(frame_reta), tk.Entry(frame_reta)
        tk.Label(frame_reta, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_reta.grid(row=0, column=1)
        tk.Label(frame_reta, text="x1:").grid(row=1, column=0, sticky="w", pady=2)
        x1.grid(row=1, column=1)
        tk.Label(frame_reta, text="y1:").grid(row=2, column=0, sticky="w", pady=2)
        y1.grid(row=2, column=1)
        tk.Label(frame_reta, text="x2:").grid(row=3, column=0, sticky="w", pady=2)
        x2.grid(row=3, column=1)
        tk.Label(frame_reta, text="y2:").grid(row=4, column=0, sticky="w", pady=2)
        y2.grid(row=4, column=1)
        tk.Button(frame_reta, text="Adicionar", command=lambda: self.add_obj(nome_reta.get(), TipoObjeto.RETA.value, [(float(x1.get()), float(y1.get())), (float(x2.get()), float(y2.get()))], popup, self.selected_color)).grid(row=5, columnspan=2, pady=10)

        # Aba Polígono
        frame_poli = ttk.Frame(notebook)
        notebook.add(frame_poli, text="Polígono")
        nome_poli = tk.Entry(frame_poli)
        coords_entry_poli = tk.Entry(frame_poli, width=30)
        preenchido_var = tk.BooleanVar()
        tk.Label(frame_poli, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_poli.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_poli, text="Coordenadas:").grid(row=1, column=0, columnspan=2, sticky="w")
        coords_entry_poli.grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")
        tk.Label(frame_poli, text="Formato: (x1,y1),(x2,y2),...").grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(frame_poli, text="Preenchido", variable=preenchido_var).grid(row=4, columnspan=2, pady=5)
        tk.Button(frame_poli, text="Adicionar", command=lambda: self.add_obj(nome_poli.get(), TipoObjeto.POLIGONO.value, list(ast.literal_eval(coords_entry_poli.get())), popup, self.selected_color, preenchido_var.get())).grid(row=5, columnspan=2, pady=10)

        # Aba Curva (Bézier)
        frame_curva_bezier = ttk.Frame(notebook)
        notebook.add(frame_curva_bezier, text="Curva Bézier")
        nome_curva_bezier = tk.Entry(frame_curva_bezier)
        coords_entry_curva_bezier = tk.Entry(frame_curva_bezier, width=30)
        tk.Label(frame_curva_bezier, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_curva_bezier.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_curva_bezier, text="Pontos de Controle:").grid(row=1, column=0, columnspan=2, sticky="w")
        coords_entry_curva_bezier.grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")
        tk.Label(frame_curva_bezier, text="Formato: (x1,y1),(x2,y2),...").grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Label(frame_curva_bezier, text="Nº de pontos: 4, 7, 10, ...").grid(row=4, column=0, columnspan=2, sticky="w", pady=4)
        tk.Button(frame_curva_bezier, text="Adicionar", command=lambda: self.add_obj(nome_curva_bezier.get(), TipoObjeto.CURVA.value, list(ast.literal_eval(coords_entry_curva_bezier.get())), popup, self.selected_color)).grid(row=5, columnspan=2, pady=10)
        
        # Aba B-Spline
        frame_bspline = ttk.Frame(notebook)
        notebook.add(frame_bspline, text="B-Spline")
        nome_bspline = tk.Entry(frame_bspline)
        coords_entry_bspline = tk.Entry(frame_bspline, width=30)
        tk.Label(frame_bspline, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_bspline.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_bspline, text="Pontos de Controle:").grid(row=1, column=0, columnspan=2, sticky="w")
        coords_entry_bspline.grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")
        tk.Label(frame_bspline, text="Formato: (x1,y1),(x2,y2),...").grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Label(frame_bspline, text="Mínimo de 4 pontos.").grid(row=4, column=0, columnspan=2, sticky="w", pady=4)
        tk.Button(frame_bspline, text="Adicionar", command=lambda: self.add_obj(nome_bspline.get(), TipoObjeto.B_SPLINE.value, list(ast.literal_eval(coords_entry_bspline.get())), popup, self.selected_color)).grid(row=5, columnspan=2, pady=10)

        # ================== ABA OBJETO 3D ==================
        frame_obj3d = ttk.Frame(notebook)
        notebook.add(frame_obj3d, text="Objeto 3D")
        
        nome_obj3d = tk.Entry(frame_obj3d)
        tk.Label(frame_obj3d, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_obj3d.grid(row=0, column=1, sticky="ew")

        tk.Label(frame_obj3d, text="Vértices (x,y,z):").grid(row=1, column=0, sticky="w")
        vertices_text = tk.Text(frame_obj3d, height=6)
        vertices_text.grid(row=2, column=0, columnspan=2, sticky="ew")
        tk.Label(frame_obj3d, text="Formato: (x1,y1,z1); (x2,y2,z2); ...").grid(row=3, column=0, columnspan=2, sticky="w")

        tk.Label(frame_obj3d, text="Arestas (índices):").grid(row=4, column=0, sticky="w", pady=(5,0))
        arestas_text = tk.Text(frame_obj3d, height=6)
        arestas_text.grid(row=5, column=0, columnspan=2, sticky="ew")
        tk.Label(frame_obj3d, text="Formato: (0,1); (1,2); ...").grid(row=6, column=0, columnspan=2, sticky="w")

        # Exemplo de cubo
        vertices_text.insert("1.0", "(-50,-50,-50); (50,-50,-50); (50,50,-50); (-50,50,-50); (-50,-50,50); (50,-50,50); (50,50,50); (-50,50,50)")
        arestas_text.insert("1.0", "(0,1);(1,2);(2,3);(3,0);(4,5);(5,6);(6,7);(7,4);(0,4);(1,5);(2,6);(3,7)")
        
        tk.Button(frame_obj3d, text="Adicionar", 
                  command=lambda: self.add_obj_3d(nome_obj3d.get(), vertices_text.get("1.0", "end-1c"), arestas_text.get("1.0", "end-1c"), popup, self.selected_color)
        ).grid(row=7, columnspan=2, pady=10)

    def escolher_cor(self):
        cor_codigo = colorchooser.askcolor(title="Escolha uma cor")[1]
        if cor_codigo:
            self.selected_color = cor_codigo
            self.cor_preview.config(bg=cor_codigo)

    def add_obj_3d(self, nome, vertices_str, arestas_str, popup, cor):
        try:
            vertices_raw = [ast.literal_eval(v.strip()) for v in vertices_str.split(';') if v.strip()]
            vertices = [Ponto3D(x,y,z) for x,y,z in vertices_raw]
            
            arestas = [ast.literal_eval(a.strip()) for a in arestas_str.split(';') if a.strip()]

            if not vertices or not arestas: raise ValueError("Vértices ou arestas vazios.")

            obj = Objeto3D(nome if nome else f"obj3d_{len(self.displayfile.objetos)}", vertices, arestas, cor)
            self.displayfile.adicionar(obj)
            self.redesenhar()
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
            popup.destroy()
        except Exception as e:
            messagebox.showerror("Erro de Entrada", f"Dados do objeto 3D inválidos.\n\nErro: {e}")

    def add_obj(self, nome, tipo, coords, popup, cor, preenchido=False):
        try:
            if tipo == TipoObjeto.CURVA.value: # Bézier
                if len(coords) < 4 or (len(coords) - 4) % 3 != 0:
                    messagebox.showerror("Erro de Entrada", "Uma curva de Bézier precisa de 4, 7, 10, ... pontos de controle.")
                    return
            
            if tipo == TipoObjeto.B_SPLINE.value:
                if len(coords) < 4:
                    messagebox.showerror("Erro de Entrada", "Uma B-Spline precisa de no mínimo 4 pontos de controle.")
                    return
            
            coords_float = [(float(x), float(y)) for x, y in coords]
            obj = Objeto(nome if nome else f"{tipo}_{len(self.displayfile.objetos)}", tipo, coords_float, cor, preenchido)
            self.displayfile.adicionar(obj)
            self.redesenhar()
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
            popup.destroy()
        except (ValueError, TypeError, SyntaxError) as e:
            messagebox.showerror("Erro de Entrada", f"Coordenadas inválidas. Verifique o formato.\n\nErro: {e}")

    def abrir_transformacoes_popup(self):
        if not self.selected_obj_name:
            messagebox.showerror("Erro", "Selecione um objeto para transformar.")
            return

        obj = self.displayfile.get_by_name(self.selected_obj_name)
        if not obj: return
        
        if obj.tipo == TipoObjeto.OBJETO_3D.value:
            self.abrir_transformacoes_popup_3d(obj)
        else:
            self.abrir_transformacoes_popup_2d(obj)

    def abrir_transformacoes_popup_3d(self, obj):
        popup = tk.Toplevel(self)
        popup.title(f"Transformar {obj.nome}")
        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # Translação 3D
        frame_t = ttk.Frame(notebook)
        notebook.add(frame_t, text="Translação")

        dx_t, dy_t, dz_t = tk.Entry(frame_t), tk.Entry(frame_t), tk.Entry(frame_t)
        tk.Label(frame_t, text="dx:").grid(row=0, column=0)
        dx_t.grid(row=0, column=1)
        tk.Label(frame_t, text="dy:").grid(row=1, column=0)
        dy_t.grid(row=1, column=1)
        tk.Label(frame_t, text="dz:").grid(row=2, column=0)
        dz_t.grid(row=2, column=1)

        tk.Button(frame_t, text="Aplicar", command=lambda: self.aplicar_transformacao_3d(
            obj, "translacao", dx=float(dx_t.get() or 0), dy=float(dy_t.get() or 0), dz=float(dz_t.get() or 0), popup=popup)).grid(row=3, columnspan=2)

        # Escalonamento 3D
        frame_s = ttk.Frame(notebook)
        notebook.add(frame_s, text="Escalonamento")

        sx_s, sy_s, sz_s = tk.Entry(frame_s), tk.Entry(frame_s), tk.Entry(frame_s)
        tk.Label(frame_s, text="sx:").grid(row=0, column=0)
        sx_s.grid(row=0, column=1)
        tk.Label(frame_s, text="sy:").grid(row=1, column=0)
        sy_s.grid(row=1, column=1)
        tk.Label(frame_s, text="sz:").grid(row=2, column=0)
        sz_s.grid(row=2, column=1)
        tk.Button(frame_s, text="Aplicar", command=lambda: self.aplicar_transformacao_3d(
            obj, "escalonamento", sx=float(sx_s.get() or 1), sy=float(sy_s.get() or 1), sz=float(sz_s.get() or 1), popup=popup)).grid(row=3, columnspan=2)

        # Rotação 3D
        frame_r = ttk.Frame(notebook)
        notebook.add(frame_r, text="Rotação")
        angulo_r = tk.Entry(frame_r)
        eixo_var = tk.StringVar(value="y")
        tk.Label(frame_r, text="Ângulo (graus):").grid(row=0, column=0)
        angulo_r.grid(row=0, column=1)
        tk.Label(frame_r, text="Eixo:").grid(row=1, column=0)
        tk.Radiobutton(frame_r, text="X", variable=eixo_var, value="x").grid(row=1, column=1, sticky="w")
        tk.Radiobutton(frame_r, text="Y", variable=eixo_var, value="y").grid(row=2, column=1, sticky="w")
        tk.Radiobutton(frame_r, text="Z", variable=eixo_var, value="z").grid(row=3, column=1, sticky="w")
        tk.Button(frame_r, text="Aplicar", command=lambda: self.aplicar_transformacao_3d(
            obj, "rotacao", angulo=float(angulo_r.get() or 0), eixo=eixo_var.get(), popup=popup)).grid(row=4, columnspan=2)


    def abrir_transformacoes_popup_2d(self):
        if not self.selected_obj_name:
            messagebox.showerror("Erro", "Selecione um objeto para transformar.")
            return

        self.lista_objetos.unbind("<<ListboxSelect>>")
        popup = tk.Toplevel(self)
        popup.title(f"Transformar {self.selected_obj_name}")
        popup.geometry("350x300")
        popup.protocol("WM_DELETE_WINDOW", lambda: self.fechar_popup_transformacao(popup))
        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)

        frame_t = ttk.Frame(notebook)
        notebook.add(frame_t, text="Translação")
        dx_t, dy_t = tk.Entry(frame_t), tk.Entry(frame_t)
        tk.Label(frame_t, text="dx:").grid(row=0, column=0, sticky="w")
        dx_t.grid(row=0, column=1)
        tk.Label(frame_t, text="dy:").grid(row=1, column=0, sticky="w")
        dy_t.grid(row=1, column=1)
        tk.Button(frame_t, text="Aplicar", command=lambda: self.aplicar_transformacao("translacao", dx=float(dx_t.get() or 0), dy=float(dy_t.get() or 0), popup=popup)).grid(row=2, columnspan=2, pady=10)

        frame_s = ttk.Frame(notebook)
        notebook.add(frame_s, text="Escalonamento")
        sx_s, sy_s = tk.Entry(frame_s), tk.Entry(frame_s)
        tk.Label(frame_s, text="sx:").grid(row=0, column=0, sticky="w")
        sx_s.grid(row=0, column=1)
        tk.Label(frame_s, text="sy:").grid(row=1, column=0, sticky="w")
        sy_s.grid(row=1, column=1)
        tk.Button(frame_s, text="Aplicar", command=lambda: self.aplicar_transformacao("escalonamento_natural", sx=float(sx_s.get() or 1), sy=float(sy_s.get() or 1), popup=popup)).grid(row=2, columnspan=2, pady=10)

        frame_r = ttk.Frame(notebook)
        notebook.add(frame_r, text="Rotação")
        tk.Label(frame_r, text="Ângulo (graus):").grid(row=0, column=0, sticky="w")
        angulo_r = tk.Entry(frame_r)
        angulo_r.grid(row=0, column=1)
        rot_var = tk.StringVar(value="centro_objeto")
        tk.Radiobutton(frame_r, text="Centro do objeto", variable=rot_var, value="centro_objeto").grid(row=1, sticky="w")
        tk.Radiobutton(frame_r, text="Origem do mundo (0,0)", variable=rot_var, value="centro_mundo").grid(row=2, sticky="w")
        frame_arbitrario = tk.Frame(frame_r)
        frame_arbitrario.grid(row=3, sticky="w")
        tk.Radiobutton(frame_arbitrario, text="Ponto:", variable=rot_var, value="ponto_arbitrario").pack(side="left")
        x_r, y_r = tk.Entry(frame_arbitrario, width=5), tk.Entry(frame_arbitrario, width=5)
        tk.Label(frame_arbitrario, text="x:").pack(side="left")
        x_r.pack(side="left")
        tk.Label(frame_arbitrario, text="y:").pack(side="left")
        y_r.pack(side="left")
        tk.Button(frame_r, text="Aplicar", command=lambda: self.aplicar_transformacao(rot_var.get(), angulo=float(angulo_r.get() or 0), px=float(x_r.get() or 0), py=float(y_r.get() or 0), popup=popup)).grid(row=4, columnspan=2, pady=10)

    def aplicar_transformacao_3d(self, obj, tipo_transf, **kwargs):
        popup = kwargs.pop("popup", None)
        try:
            if tipo_transf == "translacao":
                matriz = Transformacoes3D.get_matriz_translacao(kwargs["dx"], kwargs["dy"], kwargs["dz"])
                obj.aplicar_transformacao_3d(matriz)
            elif tipo_transf == "escalonamento":
                cx, cy, cz = obj.get_centro_objeto_3d()
                t1 = Transformacoes3D.get_matriz_translacao(-cx, -cy, -cz)
                s = Transformacoes3D.get_matriz_escalonamento(kwargs["sx"], kwargs["sy"], kwargs["sz"])
                t2 = Transformacoes3D.get_matriz_translacao(cx, cy, cz)
                matriz = t2 @ s @ t1
                obj.aplicar_transformacao_3d(matriz)
            elif tipo_transf == "rotacao":
                angulo_rad = np.deg2rad(kwargs["angulo"])
                eixo = kwargs["eixo"]
                cx, cy, cz = obj.get_centro_objeto_3d()
                t1 = Transformacoes3D.get_matriz_translacao(-cx, -cy, -cz)
                if eixo == 'x': r = Transformacoes3D.get_matriz_rotacao_x(angulo_rad)
                elif eixo == 'y': r = Transformacoes3D.get_matriz_rotacao_y(angulo_rad)
                else: r = Transformacoes3D.get_matriz_rotacao_z(angulo_rad)
                t2 = Transformacoes3D.get_matriz_translacao(cx, cy, cz)
                matriz = t2 @ r @ t1
                obj.aplicar_transformacao_3d(matriz)
            
            self.redesenhar()
            if popup: popup.destroy()
        except Exception as e:
            messagebox.showerror("Erro", f"Entrada inválida: {e}")

    def aplicar_transformacao(self, tipo_transf, **kwargs):
        obj = self.displayfile.get_by_name(self.selected_obj_name)
        if not obj: return
        try:
            if tipo_transf == "translacao":
                matriz = self.transformacoes.get_matriz_translacao(kwargs["dx"], kwargs["dy"])
                self.transformacoes.aplicar_transformacao_generica(obj, matriz)
            elif tipo_transf == "escalonamento_natural":
                self.transformacoes.aplicar_escalonamento_natural(obj, kwargs["sx"], kwargs["sy"])
            elif tipo_transf == "centro_mundo":
                matriz = self.transformacoes.get_matriz_rotacao(np.deg2rad(kwargs["angulo"]))
                self.transformacoes.aplicar_transformacao_generica(obj, matriz)
            elif tipo_transf == "centro_objeto":
                self.transformacoes.aplicar_rotacao_centro_objeto(obj, np.deg2rad(kwargs["angulo"]))
            elif tipo_transf == "ponto_arbitrario":
                self.transformacoes.aplicar_rotacao_ponto_arbitrario(obj, np.deg2rad(kwargs["angulo"]), kwargs["px"], kwargs["py"])
            
            self.redesenhar()
            if "popup" in kwargs: self.fechar_popup_transformacao(kwargs["popup"])
        except (ValueError, TypeError) as e:
            messagebox.showerror("Erro de Entrada", f"Entrada inválida. Verifique os valores.\n\nErro: {e}")

    def pan(self, dx, dy):
        theta = np.deg2rad(self.viewport.window_angle)
        c, s = np.cos(theta), np.sin(theta)
        world_dx, world_dy = dx * c - dy * s, dx * s + dy * c
        self.viewport.wmin = (self.viewport.wmin[0] + world_dx, self.viewport.wmin[1] + world_dy)
        self.viewport.wmax = (self.viewport.wmax[0] + world_dx, self.viewport.wmax[1] + world_dy)
        self.redesenhar()

    def zoom(self, fator):
        cx = (self.viewport.wmin[0] + self.viewport.wmax[0]) / 2
        cy = (self.viewport.wmin[1] + self.viewport.wmax[1]) / 2
        largura = (self.viewport.wmax[0] - self.viewport.wmin[0]) * fator
        altura = (self.viewport.wmax[1] - self.viewport.wmin[1]) * fator
        self.viewport.wmin = (cx - largura / 2, cy - altura / 2)
        self.viewport.wmax = (cx + largura / 2, cy + altura / 2)
        self.redesenhar()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("SGI 2D/3D - Projeção Ortogonal")
    root.geometry("1200x800")
    app = App(root)
    app.mainloop()