# renderizacao.py
import tkinter as tk
import numpy as np
import copy
from typing import List, Tuple
from geometria_3d import Ponto3D, Transformacoes3D
from objetos import Objeto, Objeto3D, SuperficieBezier, SuperficieBSpline, TipoObjeto


class Camera:
    def __init__(self, vrp: Ponto3D, p_foco: Ponto3D, vup: tuple, d: float = 300):
        self.vrp = vrp
        self.p_foco = p_foco
        self.vup = np.array(vup)
        self.d = d
        self.matriz_wc = self._calcular_matriz_wc()

    def set_distancia_plano_projecao(self, d):
        self.d = d

    def _calcular_matriz_wc(self):
        vpn = self.p_foco.coords[:3] - self.vrp.coords[:3]

        n = Transformacoes3D._normalizar(vpn)
        u = Transformacoes3D._normalizar(np.cross(self.vup, n))
        v = Transformacoes3D._normalizar(np.cross(n, u))

        matriz_rot = np.array(
            [np.append(u, 0), np.append(v, 0), np.append(n, 0), [0, 0, 0, 1]]
        )

        matriz_trans = Transformacoes3D.get_matriz_translacao(
            -self.vrp.x, -self.vrp.y, -self.vrp.z
        )

        return matriz_rot @ matriz_trans

    def get_matriz_wc(self):
        return self.matriz_wc

    def mover(self, dx, dy, dz):
        self.vrp.aplicar_transformacao(
            Transformacoes3D.get_matriz_translacao(dx, dy, dz)
        )
        self.p_foco.aplicar_transformacao(
            Transformacoes3D.get_matriz_translacao(dx, dy, dz)
        )
        self.matriz_wc = self._calcular_matriz_wc()

    def girar(self, angulo_y_rad, angulo_x_rad):
        vpn = self.p_foco.coords[:3] - self.vrp.coords[:3]
        n = Transformacoes3D._normalizar(vpn)
        u = Transformacoes3D._normalizar(np.cross(self.vup, n))

        rot_y = Transformacoes3D.get_matriz_rotacao_y(angulo_y_rad)
        rot_x_matriz = Transformacoes3D.get_matriz_rotacao_eixo_arbitrario(
            (0, 0, 0), u, angulo_x_rad
        )

        novo_p_foco_coords = rot_y @ rot_x_matriz @ np.append(vpn, 1)

        self.p_foco = Ponto3D(
            self.vrp.x + novo_p_foco_coords[0],
            self.vrp.y + novo_p_foco_coords[1],
            self.vrp.z + novo_p_foco_coords[2],
        )
        self.matriz_wc = self._calcular_matriz_wc()

    def mover_relativo(self, d_direita, d_cima, d_frente):
        vpn = self.p_foco.coords[:3] - self.vrp.coords[:3]
        n = Transformacoes3D._normalizar(vpn)
        u = Transformacoes3D._normalizar(np.cross(self.vup, n))
        v = Transformacoes3D._normalizar(np.cross(n, u))

        movimento_world = (d_direita * -u) + (d_cima * v) + (d_frente * n)

        matriz_movimento = Transformacoes3D.get_matriz_translacao(
            movimento_world[0], movimento_world[1], movimento_world[2]
        )

        self.vrp.aplicar_transformacao(matriz_movimento)
        self.p_foco.aplicar_transformacao(matriz_movimento)
        self.matriz_wc = self._calcular_matriz_wc()


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
    M_BSPLINE = (1 / 6) * np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]
    )

    @staticmethod
    def _get_matriz_E(delta: float) -> np.ndarray:
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

    def gerar_segmentos(
        self, pontos_controle: List[Tuple[float, float]], delta: float = 0.1
    ) -> List[List[Tuple[float, float]]]:
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
    INSIDE = 0
    LEFT = 1
    RIGHT = 2
    BOTTOM = 4
    TOP = 8

    def _get_outcode(self, x, y, wmin, wmax):
        code = self.INSIDE
        if x < wmin[0]:
            code |= self.LEFT
        elif x > wmax[0]:
            code |= self.RIGHT
        if y < wmin[1]:
            code |= self.BOTTOM
        elif y > wmax[1]:
            code |= self.TOP
        return code

    def cohen_sutherland(self, p1, p2, wmin, wmax):
        x1, y1 = p1
        x2, y2 = p2
        outcode1 = self._get_outcode(x1, y1, wmin, wmax)
        outcode2 = self._get_outcode(x2, y2, wmin, wmax)

        while True:
            if not (outcode1 | outcode2):
                return [(x1, y1), (x2, y2)]
            elif outcode1 & outcode2:
                return []
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

        if u1 > u2:
            return []

        clipped_p1 = (x1 + u1 * dx, y1 + u1 * dy)
        clipped_p2 = (x1 + u2 * dx, y1 + u2 * dy)
        return [clipped_p1, clipped_p2]

    def sutherland_hodgman(self, vertices, wmin, wmax):
        def clip_contra_aresta(verts, aresta):
            novos_verts = []
            for i in range(len(verts)):
                p1, p2 = verts[i], verts[(i + 1) % len(verts)]
                inside1, inside2 = esta_dentro(p1, aresta), esta_dentro(p2, aresta)
                if inside1 and inside2:
                    novos_verts.append(p2)
                elif inside1 and not inside2:
                    novos_verts.append(intersecao(p1, p2, aresta))
                elif not inside1 and inside2:
                    novos_verts.append(intersecao(p1, p2, aresta))
                    novos_verts.append(p2)
            return novos_verts

        def esta_dentro(p, aresta):
            if aresta == "left":
                return p[0] >= wmin[0]
            if aresta == "right":
                return p[0] <= wmax[0]
            if aresta == "bottom":
                return p[1] >= wmin[1]
            if aresta == "top":
                return p[1] <= wmax[1]

        def intersecao(p1, p2, aresta):
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1
            if aresta == "left":
                y = y1 + dy * (wmin[0] - x1) / dx if dx != 0 else y1
                return (wmin[0], y)
            if aresta == "right":
                y = y1 + dy * (wmax[0] - x1) / dx if dx != 0 else y1
                return (wmax[0], y)
            if aresta == "bottom":
                x = x1 + dx * (wmin[1] - y1) / dy if dy != 0 else x1
                return (x, wmin[1])
            if aresta == "top":
                x = x1 + dx * (wmax[1] - y1) / dy if dy != 0 else x1
                return (x, wmax[1])

        clipped_vertices = vertices
        for aresta in ["left", "right", "bottom", "top"]:
            clipped_vertices = clip_contra_aresta(clipped_vertices, aresta)
        return clipped_vertices

    def _gerar_pontos_bezier(self, pontos_controle, num_passos=20):
        pontos = []
        p1, p2, p3, p4 = [np.array(p) for p in pontos_controle]
        for i in range(num_passos + 1):
            t = i / num_passos
            ponto = (
                (1 - t) ** 3 * p1
                + 3 * (1 - t) ** 2 * t * p2
                + 3 * (1 - t) * t**2 * p3
                + t**3 * p4
            )
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
            linha = self.cohen_sutherland(
                pontos_controle[0], pontos_controle[-1], wmin, wmax
            )
            return [linha] if linha else []

        todos_dentro = all(
            self._get_outcode(p[0], p[1], wmin, wmax) == self.INSIDE
            for p in pontos_controle
        )
        if todos_dentro:
            pontos_gerados = self._gerar_pontos_bezier(pontos_controle)
            segmentos = []
            for i in range(len(pontos_gerados) - 1):
                segmentos.append([pontos_gerados[i], pontos_gerados[i + 1]])
            return segmentos

        outcode_comum = 0b1111
        for p in pontos_controle:
            outcode_comum &= self._get_outcode(p[0], p[1], wmin, wmax)
        if outcode_comum != self.INSIDE:
            return []

        curva1, curva2 = self._subdivide_bezier(pontos_controle, 0.5)

        segmentos_clipados = []
        segmentos_clipados.extend(
            self._clip_bezier_recursivo(curva1, wmin, wmax, profundidade + 1)
        )
        segmentos_clipados.extend(
            self._clip_bezier_recursivo(curva2, wmin, wmax, profundidade + 1)
        )
        return segmentos_clipados

    def clip_curva_para_linhas(self, pontos_controle_total, wmin, wmax):
        segmentos_de_linha_visiveis = []
        if len(pontos_controle_total) < 4:
            return []

        num_segmentos = (len(pontos_controle_total) - 4) // 3 + 1

        for i in range(num_segmentos):
            inicio_idx = i * 3
            pontos_segmento_atual = pontos_controle_total[inicio_idx : inicio_idx + 4]
            linhas_clipadas = self._clip_bezier_recursivo(
                pontos_segmento_atual, wmin, wmax
            )
            segmentos_de_linha_visiveis.extend(linhas_clipadas)

        return segmentos_de_linha_visiveis

    def clip(self, obj, wmin, wmax, alg_reta):
        coords_clipadas = []
        if obj.tipo == TipoObjeto.PONTO.value:
            x, y = obj.coords[0]
            if wmin[0] <= x <= wmax[0] and wmin[1] <= y <= wmax[1]:
                coords_clipadas = obj.coords

        elif obj.tipo == TipoObjeto.RETA.value:
            if alg_reta == "cs":
                coords_clipadas = self.cohen_sutherland(
                    obj.coords[0], obj.coords[1], wmin, wmax
                )
            else:
                coords_clipadas = self.liang_barsky(
                    obj.coords[0], obj.coords[1], wmin, wmax
                )

        elif obj.tipo == TipoObjeto.POLIGONO.value:
            coords_clipadas = self.sutherland_hodgman(obj.coords, wmin, wmax)

        if not coords_clipadas:
            return None

        if obj.tipo in [
            TipoObjeto.CURVA.value,
            TipoObjeto.B_SPLINE.value,
            TipoObjeto.OBJETO_3D.value,
            TipoObjeto.SUPERFICIE_BEZIER.value,
            TipoObjeto.SUPERFICIE_BSPLINE.value,
        ]:
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

        if xw_range == 0 or yw_range == 0:
            return (0, 0)

        xvp = vpmin[0] + ((xw - w_min[0]) / xw_range) * xvp_range
        yvp = vpmin[1] + (1 - (yw - w_min[1]) / yw_range) * yvp_range
        return (xvp, yvp)

    def desenhar(
        self, displayfile: DisplayFile, alg_reta_clip: str, tipo_projecao: str
    ):
        self.canvas.delete("all")
        canvas_width, canvas_height = (
            self.canvas.winfo_width(),
            self.canvas.winfo_height(),
        )
        canvas_margin = 10
        if canvas_width <= (2 * canvas_margin) or canvas_height <= (2 * canvas_margin):
            return

        available_width, available_height = canvas_width - (
            2 * canvas_margin
        ), canvas_height - (2 * canvas_margin)
        w_width, w_height = self.wmax[0] - self.wmin[0], self.wmax[1] - self.wmin[1]

        if w_height == 0:
            w_height = 1e-9
        
        w_aspect = w_width / w_height
        
        if available_height == 0:
            available_height = 1e-9
            
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

        self.canvas.create_rectangle(
            vpmin[0], vpmin[1], vpmax[0], vpmax[1], outline="blue", dash=(4, 4)
        )

        cx, cy = (self.wmin[0] + self.wmax[0]) / 2.0, (
            self.wmin[1] + self.wmax[1]
        ) / 2.0
        ang_rad = -np.deg2rad(self.window_angle)
        
        axis_len = max(abs(self.wmin[0]), abs(self.wmax[0]), abs(self.wmin[1]), abs(self.wmax[1])) * 10
        
        p_x_start_world = (-axis_len, 0)
        p_x_end_world = (axis_len, 0)
        p_y_start_world = (0, -axis_len)
        p_y_end_world = (0, axis_len)

        p_x_start_r = self._rotacionar_ponto_em_torno_de(p_x_start_world[0], p_x_start_world[1], cx, cy, ang_rad)
        p_x_end_r = self._rotacionar_ponto_em_torno_de(p_x_end_world[0], p_x_end_world[1], cx, cy, ang_rad)
        p_y_start_r = self._rotacionar_ponto_em_torno_de(p_y_start_world[0], p_y_start_world[1], cx, cy, ang_rad)
        p_y_end_r = self._rotacionar_ponto_em_torno_de(p_y_end_world[0], p_y_end_world[1], cx, cy, ang_rad)

        x_axis_clipped = self.clipping.cohen_sutherland(p_x_start_r, p_x_end_r, self.wmin, self.wmax)
        y_axis_clipped = self.clipping.cohen_sutherland(p_y_start_r, p_y_end_r, self.wmin, self.wmax)

        if x_axis_clipped:
            p1_vp = self.transformar(x_axis_clipped[0][0], x_axis_clipped[0][1], self.wmin, self.wmax, vpmin, vpmax)
            p2_vp = self.transformar(x_axis_clipped[1][0], x_axis_clipped[1][1], self.wmin, self.wmax, vpmin, vpmax)
            self.canvas.create_line(p1_vp, p2_vp, fill="gray", dash=(2, 2))
        
        if y_axis_clipped:
            p1_vp = self.transformar(y_axis_clipped[0][0], y_axis_clipped[0][1], self.wmin, self.wmax, vpmin, vpmax)
            p2_vp = self.transformar(y_axis_clipped[1][0], y_axis_clipped[1][1], self.wmin, self.wmax, vpmin, vpmax)
            self.canvas.create_line(p1_vp, p2_vp, fill="gray", dash=(2, 2))


        for obj in displayfile.listar():
            if isinstance(obj, (Objeto3D, SuperficieBezier, SuperficieBSpline)):
                obj_camera_coords = copy.deepcopy(obj)
                matriz_wc = self.camera.get_matriz_wc()
                obj_camera_coords.aplicar_transformacao_3d(matriz_wc)

                arestas_para_processar = []
                if obj.tipo == TipoObjeto.OBJETO_3D.value:
                    for idx1, idx2 in obj_camera_coords.arestas:
                        if 0 <= idx1 < len(obj_camera_coords.vertices) and 0 <= idx2 < len(obj_camera_coords.vertices):
                            arestas_para_processar.append(
                                (
                                    obj_camera_coords.vertices[idx1],
                                    obj_camera_coords.vertices[idx2],
                                )
                            )
                elif obj.tipo in (
                    TipoObjeto.SUPERFICIE_BEZIER.value,
                    TipoObjeto.SUPERFICIE_BSPLINE.value,
                ):
                    arestas_para_processar = obj_camera_coords.gerar_malha()

                for p1_cam, p2_cam in arestas_para_processar:
                    if (p1_cam.z <= 1 or p2_cam.z <= 1):
                        continue

                    if tipo_projecao == "perspectiva":
                        d = self.camera.d

                        x1_p = d * p1_cam.x / p1_cam.z
                        y1_p = d * p1_cam.y / p1_cam.z
                        x2_p = d * p2_cam.x / p2_cam.z
                        y2_p = d * p2_cam.y / p2_cam.z
                        p1_2d, p2_2d = (x1_p, y1_p), (x2_p, y2_p)
                    else:
                        p1_2d, p2_2d = (p1_cam.x, p1_cam.y), (p2_cam.x, p2_cam.y)

                    p1_r = self._rotacionar_ponto_em_torno_de(p1_2d[0], p1_2d[1], cx, cy, ang_rad)
                    p2_r = self._rotacionar_ponto_em_torno_de(p2_2d[0], p2_2d[1], cx, cy, ang_rad)

                    reta_clipada_coords = self.clipping.cohen_sutherland(
                        p1_r, p2_r, self.wmin, self.wmax
                    )
                    
                    if reta_clipada_coords:
                        p1_vp = self.transformar(
                            reta_clipada_coords[0][0], reta_clipada_coords[0][1], self.wmin, self.wmax, vpmin, vpmax
                        )
                        p2_vp = self.transformar(
                            reta_clipada_coords[1][0], reta_clipada_coords[1][1], self.wmin, self.wmax, vpmin, vpmax
                        )
                        linha_width = 2 if obj.tipo == TipoObjeto.OBJETO_3D.value else 1
                        self.canvas.create_line(
                            p1_vp, p2_vp, fill=obj.cor, width=linha_width
                        )
                continue

            if obj.tipo == TipoObjeto.CURVA.value:
                pontos_controle_total = obj.coords
                segmentos_de_linha = []
                if len(pontos_controle_total) >= 4:
                    num_segmentos = (len(pontos_controle_total) - 4) // 3 + 1
                    for i in range(num_segmentos):
                        inicio_idx = i * 3
                        pontos_segmento_atual = pontos_controle_total[inicio_idx : inicio_idx + 4]
                        
                        pontos_gerados = self.clipping._gerar_pontos_bezier(pontos_segmento_atual)
                        for j in range(len(pontos_gerados) - 1):
                            segmentos_de_linha.append([pontos_gerados[j], pontos_gerados[j+1]])

                for p1, p2 in segmentos_de_linha:
                    p1_r = self._rotacionar_ponto_em_torno_de(p1[0], p1[1], cx, cy, ang_rad)
                    p2_r = self._rotacionar_ponto_em_torno_de(p2[0], p2[1], cx, cy, ang_rad)
                    
                    reta_clipada = self.clipping.cohen_sutherland(p1_r, p2_r, self.wmin, self.wmax)
                    
                    if reta_clipada:
                        p1_vp = self.transformar(reta_clipada[0][0], reta_clipada[0][1], self.wmin, self.wmax, vpmin, vpmax)
                        p2_vp = self.transformar(reta_clipada[1][0], reta_clipada[1][1], self.wmin, self.wmax, vpmin, vpmax)
                        self.canvas.create_line(p1_vp, p2_vp, fill=obj.cor, width=2)
                continue

            elif obj.tipo == TipoObjeto.B_SPLINE.value:
                segmentos_de_reta = self.bspline_generator.gerar_segmentos(obj.coords)
                for p1, p2 in segmentos_de_reta:
                    p1_r = self._rotacionar_ponto_em_torno_de(p1[0], p1[1], cx, cy, ang_rad)
                    p2_r = self._rotacionar_ponto_em_torno_de(p2[0], p2[1], cx, cy, ang_rad)
                    
                    reta_clipada_coords = self.clipping.cohen_sutherland(
                        p1_r, p2_r, self.wmin, self.wmax
                    )
                    
                    if reta_clipada_coords:
                        p1_vp = self.transformar(reta_clipada_coords[0][0], reta_clipada_coords[0][1], self.wmin, self.wmax, vpmin, vpmax)
                        p2_vp = self.transformar(reta_clipada_coords[1][0], reta_clipada_coords[1][1], self.wmin, self.wmax, vpmin, vpmax)
                        self.canvas.create_line(p1_vp, p2_vp, fill=obj.cor, width=2)
                continue

            if obj.tipo == TipoObjeto.PONTO.value:
                x, y = obj.coords[0]
                xr, yr = self._rotacionar_ponto_em_torno_de(x, y, cx, cy, ang_rad)
                
                if (self.wmin[0] <= xr <= self.wmax[0] and self.wmin[1] <= yr <= self.wmax[1]):
                    x_vp, y_vp = self.transformar(xr, yr, self.wmin, self.wmax, vpmin, vpmax)
                    self.canvas.create_oval(
                        x_vp - 2, y_vp - 2, x_vp + 2, y_vp + 2,
                        fill=obj.cor, outline=obj.cor
                    )
            
            elif obj.tipo == TipoObjeto.RETA.value:
                p1, p2 = obj.coords[0], obj.coords[1]
                p1_r = self._rotacionar_ponto_em_torno_de(p1[0], p1[1], cx, cy, ang_rad)
                p2_r = self._rotacionar_ponto_em_torno_de(p2[0], p2[1], cx, cy, ang_rad)
                
                clip_alg = self.clipping.cohen_sutherland if alg_reta_clip == "cs" else self.clipping.liang_barsky
                reta_clipada = clip_alg(p1_r, p2_r, self.wmin, self.wmax)
                
                if reta_clipada:
                    p1_vp = self.transformar(reta_clipada[0][0], reta_clipada[0][1], self.wmin, self.wmax, vpmin, vpmax)
                    p2_vp = self.transformar(reta_clipada[1][0], reta_clipada[1][1], self.wmin, self.wmax, vpmin, vpmax)
                    self.canvas.create_line(p1_vp, p2_vp, fill=obj.cor, width=2)

            elif obj.tipo == TipoObjeto.POLIGONO.value:
                coords_rotadas = []
                for x, y in obj.coords:
                    coords_rotadas.append(self._rotacionar_ponto_em_torno_de(x, y, cx, cy, ang_rad))
                
                coords_clipadas = self.clipping.sutherland_hodgman(coords_rotadas, self.wmin, self.wmax)
                
                if coords_clipadas and len(coords_clipadas) > 1:
                    coords_vp = [self.transformar(x, y, self.wmin, self.wmax, vpmin, vpmax) for (x, y) in coords_clipadas]
                    
                    fill_color = obj.cor if obj.preenchido else ""
                    outline_color = "" if obj.preenchido else obj.cor
                    self.canvas.create_polygon(
                        coords_vp, fill=fill_color, outline=outline_color, width=2
                    )