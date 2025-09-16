import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
from typing import List, Tuple, Optional
from enum import Enum
import numpy as np
import ast


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
    WIREFRAME = "wireframe"


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
        self.preenchido = preenchido  # para polígonos


class DisplayFile:
    def __init__(self):
        self.objetos = []

    def adicionar(self, obj: Objeto):
        self.objetos.append(obj)

    def listar(self):
        return self.objetos

    def get_by_name(self, nome):
        for obj in self.objetos:
            if obj.nome == nome:
                return obj
        return None

    def get_object_by_coords(self, x, y):
        for obj in reversed(self.objetos):
            x_coords = [p[0] for p in obj.coords]
            y_coords = [p[1] for p in obj.coords]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            margem = 5

            if (
                x_min - margem <= x <= x_max + margem
                and y_min - margem <= y <= y_max + margem
            ):
                return obj
        return None


# =========================
# Funções de Clipagem
# =========================

def point_in_window(px, py, wmin, wmax):
    return (wmin[0] <= px <= wmax[0]) and (wmin[1] <= py <= wmax[1])


# Cohen-Sutherland constants
INSIDE = 0  # 0000
LEFT = 1    # 0001
RIGHT = 2   # 0010
BOTTOM = 4  # 0100
TOP = 8     # 1000


def compute_out_code(x, y, wmin, wmax):
    code = INSIDE
    if x < wmin[0]:
        code |= LEFT
    elif x > wmax[0]:
        code |= RIGHT
    if y < wmin[1]:
        code |= BOTTOM
    elif y > wmax[1]:
        code |= TOP
    return code


def cohen_sutherland_clip(p1, p2, wmin, wmax) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    x1, y1 = p1
    x2, y2 = p2
    out1 = compute_out_code(x1, y1, wmin, wmax)
    out2 = compute_out_code(x2, y2, wmin, wmax)
    accept = False

    while True:
        if not (out1 | out2):
            accept = True
            break
        elif out1 & out2:
            break
        else:
            # pick one endpoint outside
            outcode_out = out1 if out1 != 0 else out2
            if outcode_out & TOP:
                x = x1 + (x2 - x1) * (wmax[1] - y1) / (y2 - y1) if y2 != y1 else float('inf')
                y = wmax[1]
            elif outcode_out & BOTTOM:
                x = x1 + (x2 - x1) * (wmin[1] - y1) / (y2 - y1) if y2 != y1 else float('inf')
                y = wmin[1]
            elif outcode_out & RIGHT:
                y = y1 + (y2 - y1) * (wmax[0] - x1) / (x2 - x1) if x2 != x1 else float('inf')
                x = wmax[0]
            elif outcode_out & LEFT:
                y = y1 + (y2 - y1) * (wmin[0] - x1) / (x2 - x1) if x2 != x1 else float('inf')
                x = wmin[0]
            else:
                break

            if outcode_out == out1:
                x1, y1 = x, y
                out1 = compute_out_code(x1, y1, wmin, wmax)
            else:
                x2, y2 = x, y
                out2 = compute_out_code(x2, y2, wmin, wmax)

    if accept:
        return ((x1, y1), (x2, y2))
    else:
        return None


def liang_barsky_clip(p1, p2, wmin, wmax) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    p = [-dx, dx, -dy, dy]
    q = [x1 - wmin[0], wmax[0] - x1, y1 - wmin[1], wmax[1] - y1]

    u1 = 0.0
    u2 = 1.0

    for pk, qk in zip(p, q):
        if pk == 0:
            if qk < 0:
                return None  # paralelo e fora
            else:
                continue
        r = qk / pk
        if pk < 0:
            if r > u2:
                return None
            if r > u1:
                u1 = r
        else:
            if r < u1:
                return None
            if r < u2:
                u2 = r

    if u1 > u2:
        return None

    nx1 = x1 + u1 * dx
    ny1 = y1 + u1 * dy
    nx2 = x1 + u2 * dx
    ny2 = y1 + u2 * dy

    return ((nx1, ny1), (nx2, ny2))


# Sutherland-Hodgman polygon clipping against axis-aligned rectangle
def sutherland_hodgman_clip(polygon: List[Tuple[float, float]], wmin, wmax) -> List[Tuple[float, float]]:
    def clip_edge(poly, edge):
        clipped = []
        for i in range(len(poly)):
            curr = poly[i]
            prev = poly[i - 1]
            inside_curr = edge_inside(curr, edge)
            inside_prev = edge_inside(prev, edge)
            if inside_curr:
                if not inside_prev:
                    inter = compute_intersection(prev, curr, edge)
                    if inter is not None:
                        clipped.append(inter)
                clipped.append(curr)
            elif inside_prev:
                inter = compute_intersection(prev, curr, edge)
                if inter is not None:
                    clipped.append(inter)
        return clipped

    def edge_inside(pt, edge):
        x, y = pt
        if edge == 'left':
            return x >= wmin[0]
        if edge == 'right':
            return x <= wmax[0]
        if edge == 'bottom':
            return y >= wmin[1]
        if edge == 'top':
            return y <= wmax[1]

    def compute_intersection(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        if edge == 'left':
            x = wmin[0]
            if dx == 0:
                return None
            t = (x - x1) / dx
            y = y1 + t * dy
            return (x, y)
        if edge == 'right':
            x = wmax[0]
            if dx == 0:
                return None
            t = (x - x1) / dx
            y = y1 + t * dy
            return (x, y)
        if edge == 'bottom':
            y = wmin[1]
            if dy == 0:
                return None
            t = (y - y1) / dy
            x = x1 + t * dx
            return (x, y)
        if edge == 'top':
            y = wmax[1]
            if dy == 0:
                return None
            t = (y - y1) / dy
            x = x1 + t * dx
            return (x, y)

    output = polygon[:]
    for edge in ['left', 'right', 'bottom', 'top']:
        output = clip_edge(output, edge)
        if not output:
            break
    return output


# =========================
# Viewport com clipping
# =========================

class Viewport:
    def __init__(self, canvas, wmin, wmax, inset=20):
        self.canvas = canvas
        self.wmin = wmin
        self.wmax = wmax
        self.window_angle = 0.0
        self.inset = inset  # margem interna em pixels para desenhar a viewport menor que o canvas

        # default algorithm for line clipping
        self.line_clip_method = 'cohen'  # 'cohen' or 'liang'

    def set_window_angle(self, ang_deg):
        self.window_angle = ang_deg

    def set_line_clip_method(self, method: str):
        if method in ('cohen', 'liang'):
            self.line_clip_method = method

    def _rotacionar_ponto_em_torno_de(self, x, y, cx, cy, ang_rad):
        tx = x - cx
        ty = y - cy
        c = np.cos(ang_rad)
        s = np.sin(ang_rad)
        xr = c * tx - s * ty
        yr = s * tx + c * ty
        return (xr + cx, yr + cy)

    def transformar(self, xw, yw, w_adj_min, w_adj_max, vpmin, vpmax):
        xw_range = w_adj_max[0] - w_adj_min[0]
        yw_range = w_adj_max[1] - w_adj_min[1]

        xvp_range = vpmax[0] - vpmin[0]
        yvp_range = vpmax[1] - vpmin[1]

        if xw_range == 0 or yw_range == 0:
            return (0, 0)

        xvp = vpmin[0] + ((xw - w_adj_min[0]) / xw_range) * xvp_range
        yvp = vpmin[1] + (1 - (yw - w_adj_min[1]) / yw_range) * yvp_range

        return (xvp, yvp)

    def desenhar(self, displayfile: DisplayFile):
        self.canvas.delete("all")

        vp_width = self.canvas.winfo_width()
        vp_height = self.canvas.winfo_height()
        if vp_width == 0 or vp_height == 0:
            return

        # vpmin/vpmax com margem interna para deixar viewport menor que canvas
        margin = self.inset
        vpmin = (margin, margin)
        vpmax = (vp_width - margin, vp_height - margin)

        # desenhar moldura da viewport para visualização
        self.canvas.create_rectangle(vpmin[0]-1, vpmin[1]-1, vpmax[0]+1, vpmax[1]+1, outline='black', width=1)

        vp_aspect = (vpmax[0] - vpmin[0]) / (vpmax[1] - vpmin[1])
        w_width = self.wmax[0] - self.wmin[0]
        w_height = self.wmax[1] - self.wmin[1]
        w_aspect = w_width / w_height if w_height != 0 else 1

        w_adj_min = list(self.wmin)
        w_adj_max = list(self.wmax)

        if w_aspect < vp_aspect:
            new_width = w_height * vp_aspect
            delta_w = new_width - w_width
            w_adj_min[0] -= delta_w / 2
            w_adj_max[0] += delta_w / 2
        elif w_aspect > vp_aspect:
            new_height = w_width / vp_aspect
            delta_h = new_height - w_height
            w_adj_min[1] -= delta_h / 2
            w_adj_max[1] += delta_h / 2

        w_adj_min = tuple(w_adj_min)
        w_adj_max = tuple(w_adj_max)

        # centro da window no sistema world coordinates
        cx = (w_adj_min[0] + w_adj_max[0]) / 2.0
        cy = (w_adj_min[1] + w_adj_max[1]) / 2.0

        # ângulo a aplicar ao mundo = -window_angle (em radianos)
        ang_rad = -np.deg2rad(self.window_angle)

        # função auxiliar: rotaciona mundo em torno do centro da window e retorna coordenada resultante
        def rotate_world(xw, yw):
            return self._rotacionar_ponto_em_torno_de(xw, yw, cx, cy, ang_rad)

        # função de transformação final world -> viewport (após rotação)
        def world_to_vp(xw, yw):
            return self.transformar(xw, yw, w_adj_min, w_adj_max, vpmin, vpmax)

        # desenha eixos (após rotação)
        origem = world_to_vp(*rotate_world(0, 0))
        x_axis_start = world_to_vp(*rotate_world(w_adj_min[0], 0))
        x_axis_end = world_to_vp(*rotate_world(w_adj_max[0], 0))
        y_axis_start = world_to_vp(*rotate_world(0, w_adj_min[1]))
        y_axis_end = world_to_vp(*rotate_world(0, w_adj_max[1]))

        self.canvas.create_line(
            x_axis_start[0],
            origem[1],
            x_axis_end[0],
            origem[1],
            fill="gray",
            dash=(2, 2),
        )
        self.canvas.create_line(
            origem[0],
            y_axis_start[1],
            origem[0],
            y_axis_end[1],
            fill="gray",
            dash=(2, 2),
        )

        # Para clipping: vamos trabalhar em coordenadas rotacionadas do mundo (alinhadas à window)
        # w_adj_min/w_adj_max representam o retângulo axis-aligned após ajuste de aspect ratio.
        for obj in displayfile.listar():
            # rotaciona cada ponto do objeto para o sistema alinhado à window
            rotated_coords = [rotate_world(x, y) for (x, y) in obj.coords]

            # agora aplica clipping no retângulo w_adj_min -> w_adj_max (que é axis-aligned agora)
            if obj.tipo == TipoObjeto.PONTO.value:
                px, py = rotated_coords[0]
                if point_in_window(px, py, w_adj_min, w_adj_max):
                    xvp, yvp = world_to_vp(px, py)
                    self.canvas.create_oval(xvp - 2, yvp - 2, xvp + 2, yvp + 2, fill=obj.cor)

            elif obj.tipo == TipoObjeto.RETA.value:
                if len(rotated_coords) < 2:
                    continue
                p1 = rotated_coords[0]
                p2 = rotated_coords[1]
                clipped = None
                if self.line_clip_method == 'cohen':
                    clipped = cohen_sutherland_clip(p1, p2, w_adj_min, w_adj_max)
                else:
                    clipped = liang_barsky_clip(p1, p2, w_adj_min, w_adj_max)

                if clipped:
                    (cx1, cy1), (cx2, cy2) = clipped
                    vp_p1 = world_to_vp(cx1, cy1)
                    vp_p2 = world_to_vp(cx2, cy2)
                    self.canvas.create_line(vp_p1[0], vp_p1[1], vp_p2[0], vp_p2[1], fill=obj.cor, width=2)

            elif obj.tipo == TipoObjeto.WIREFRAME.value:
                # polígonos: aplicar Sutherland-Hodgman no conjunto de vértices rotacionados
                if len(rotated_coords) < 2:
                    continue
                clipped_poly = sutherland_hodgman_clip(rotated_coords, w_adj_min, w_adj_max)
                if not clipped_poly:
                    continue
                # transformar para viewport
                coords_vp = [world_to_vp(x, y) for x, y in clipped_poly]
                flat = [coord for p in coords_vp for coord in p]
                if obj.preenchido:
                    # create_polygon preenche por default; outline também definido
                    self.canvas.create_polygon(*flat, fill=obj.cor, outline='black')
                else:
                    # wireframe: desenhar arestas
                    if len(coords_vp) == 1:
                        x, y = coords_vp[0]
                        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=obj.cor)
                    else:
                        for i in range(len(coords_vp)):
                            p1 = coords_vp[i]
                            p2 = coords_vp[(i + 1) % len(coords_vp)]
                            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=obj.cor, width=2)


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        self.selected_obj_name = None
        self.transformacoes = Transformacoes()

        frame_main = tk.Frame(self)
        frame_main.pack(fill="both", expand=True)

        frame_main.grid_columnconfigure(0, weight=1)
        frame_main.grid_columnconfigure(1, weight=0)

        self.canvas = tk.Canvas(frame_main, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.lista_objetos = tk.Listbox(frame_main, width=30)
        self.lista_objetos.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.lista_objetos.bind("<<ListboxSelect>>", self.on_obj_select)
        self.lista_objetos.bind("<Button-3>", self.on_right_click_listbox)

        control_frame = tk.Frame(self)
        control_frame.pack(fill="x", padx=5, pady=5)

        self.displayfile = DisplayFile()
        self.viewport = Viewport(self.canvas, wmin=(-100, -100), wmax=(100, 100), inset=24)

        # opções de clipagem (rádio)
        clip_frame = tk.Frame(control_frame)
        clip_frame.pack(side='left', padx=(0,10))
        tk.Label(clip_frame, text="Clipagem de retas:").pack(anchor='w')
        self.clip_var = tk.StringVar(value='cohen')
        r1 = tk.Radiobutton(clip_frame, text="Cohen-Sutherland", variable=self.clip_var, value='cohen', command=self.on_clip_method_change)
        r2 = tk.Radiobutton(clip_frame, text="Liang-Barsky", variable=self.clip_var, value='liang', command=self.on_clip_method_change)
        r1.pack(anchor='w')
        r2.pack(anchor='w')

        tk.Button(
            control_frame, text="Adicionar Objeto", command=self.abrir_popup
        ).pack(side="left", padx=2)
        tk.Button(
            control_frame,
            text="Aplicar Transformação",
            command=self.abrir_transformacoes_popup,
        ).pack(side="left", padx=2)

        tk.Button(control_frame, text="←", command=lambda: self.pan(10, 0)).pack(
            side="left"
        )
        tk.Button(control_frame, text="→", command=lambda: self.pan(-10, 0)).pack(
            side="left"
        )
        tk.Button(control_frame, text="↑", command=lambda: self.pan(0, -10)).pack(
            side="left"
        )
        tk.Button(control_frame, text="↓", command=lambda: self.pan(0, 10)).pack(
            side="left"
        )
        tk.Button(control_frame, text="Zoom +", command=lambda: self.zoom(0.9)).pack(
            side="left", padx=(10, 2)
        )
        tk.Button(control_frame, text="Zoom -", command=lambda: self.zoom(1.1)).pack(
            side="left"
        )
        tk.Button(
            control_frame,
            text="Rotacionar Window",
            command=self.popup_rotacionar_window,
        ).pack(side="left", padx=(10, 2))

        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.selected_color = "black"

    def on_clip_method_change(self):
        method = self.clip_var.get()
        self.viewport.set_line_clip_method(method)
        self.viewport.desenhar(self.displayfile)

    def popup_rotacionar_window(self):
        popup = tk.Toplevel(self)
        popup.title("Rotacionar Window")
        tk.Label(popup, text="Ângulo (graus, sentido anti-horário):").pack(
            padx=8, pady=(8, 0)
        )
        ang_entry = tk.Entry(popup)
        ang_entry.pack(padx=8, pady=4)
        ang_entry.insert(0, str(self.viewport.window_angle))

        def aplicar():
            try:
                ang = float(ang_entry.get())
                self.viewport.set_window_angle(ang)
                self.viewport.desenhar(self.displayfile)
                popup.destroy()
            except Exception as e:
                messagebox.showerror("Erro", f"Valor de ângulo inválido: {e}")

        tk.Button(popup, text="Aplicar", command=aplicar).pack(pady=8)

    def on_canvas_resize(self, event):
        self.viewport.desenhar(self.displayfile)

    def on_obj_select(self, event):
        selecao = self.lista_objetos.curselection()
        if selecao:
            entry = self.lista_objetos.get(selecao[0])
            if " (" in entry:
                nome = entry.rsplit(" (", 1)[0]
            else:
                nome = entry
            self.selected_obj_name = nome
        else:
            self.selected_obj_name = None

    def on_right_click_listbox(self, event):
        index = self.lista_objetos.nearest(event.y)

        if index is not None:
            self.lista_objetos.selection_clear(0, tk.END)
            self.lista_objetos.selection_set(index)
            self.lista_objetos.activate(index)

            self.on_obj_select(event)

            self.show_context_menu(event.x, event.y)

    def show_context_menu(self, x, y):
        menu = tk.Menu(self.master, tearoff=0)
        menu.add_command(
            label="Transformar Objeto", command=self.abrir_transformacoes_popup
        )
        menu.post(
            self.lista_objetos.winfo_rootx() + x, self.lista_objetos.winfo_rooty() + y
        )

    # ==============================
    # POPUP DE INSERÇÃO DE OBJETOS
    # ==============================
    def abrir_popup(self):
        popup = tk.Toplevel(self)
        popup.title("Incluir Objeto")
        popup.geometry("360x360")

        frame_cor = tk.Frame(popup, pady=5)
        frame_cor.pack(fill="x", padx=10)
        tk.Label(frame_cor, text="Cor do Objeto:").pack(side="left")

        color_button = tk.Button(
            frame_cor, text="Escolher Cor", command=self.escolher_cor
        )
        color_button.pack(side="left", padx=5)

        self.cor_preview = tk.Canvas(
            frame_cor, width=20, height=20, bg="black", borderwidth=1, relief="solid"
        )
        self.cor_preview.pack(side="left")

        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=5)

        frame_ponto = ttk.Frame(notebook)
        notebook.add(frame_ponto, text="Ponto")
        nome_ponto = tk.Entry(frame_ponto)
        x_ponto = tk.Entry(frame_ponto)
        y_ponto = tk.Entry(frame_ponto)
        tk.Label(frame_ponto, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_ponto.grid(row=0, column=1)
        tk.Label(frame_ponto, text="x:").grid(row=1, column=0, sticky="w", pady=2)
        x_ponto.grid(row=1, column=1)
        tk.Label(frame_ponto, text="y:").grid(row=2, column=0, sticky="w", pady=2)
        y_ponto.grid(row=2, column=1)
        tk.Button(
            frame_ponto,
            text="Adicionar",
            command=lambda: self.add_obj(
                nome_ponto.get(),
                TipoObjeto.PONTO.value,
                [(float(x_ponto.get()), float(y_ponto.get()))],
                popup,
                self.selected_color,
                preenchido=False,
            ),
        ).grid(row=3, columnspan=2, pady=10)

        frame_reta = ttk.Frame(notebook)
        notebook.add(frame_reta, text="Reta")
        nome_reta = tk.Entry(frame_reta)
        x1 = tk.Entry(frame_reta)
        y1 = tk.Entry(frame_reta)
        x2 = tk.Entry(frame_reta)
        y2 = tk.Entry(frame_reta)
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
        tk.Button(
            frame_reta,
            text="Adicionar",
            command=lambda: self.add_obj(
                nome_reta.get(),
                TipoObjeto.RETA.value,
                [
                    (float(x1.get()), float(y1.get())),
                    (float(x2.get()), float(y2.get())),
                ],
                popup,
                self.selected_color,
                preenchido=False,
            ),
        ).grid(row=5, columnspan=2, pady=10)

        frame_wire = ttk.Frame(notebook)
        notebook.add(frame_wire, text="Wireframe / Polígono")
        nome_wire = tk.Entry(frame_wire)
        coords_entry = tk.Entry(frame_wire, width=40)
        preenchido_var = tk.BooleanVar(value=False)
        tk.Label(frame_wire, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_wire.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_wire, text="Coordenadas:").grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        coords_entry.grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")
        tk.Label(frame_wire, text="Formato: (x1,y1),(x2,y2),...").grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame_wire, text="Preenchido (polígono)", variable=preenchido_var).grid(row=4, column=0, columnspan=2, sticky='w')
        tk.Button(
            frame_wire,
            text="Adicionar",
            command=lambda: self.add_obj(
                nome_wire.get(),
                TipoObjeto.WIREFRAME.value,
                list(ast.literal_eval(coords_entry.get())),
                popup,
                self.selected_color,
                preenchido=preenchido_var.get(),
            ),
        ).grid(row=5, columnspan=2, pady=10)

    def escolher_cor(self):
        cor_codigo = colorchooser.askcolor(title="Escolha uma cor")[1]
        if cor_codigo:
            self.selected_color = cor_codigo
            self.cor_preview.config(bg=cor_codigo)

    def add_obj(self, nome, tipo, coords, popup, cor, preenchido=False):
        try:
            coords_float = [(float(x), float(y)) for x, y in coords]
            obj = Objeto(nome if nome else tipo, tipo, coords_float, cor, preenchido)
            self.displayfile.adicionar(obj)
            self.viewport.desenhar(self.displayfile)
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
            popup.destroy()
        except (ValueError, TypeError, SyntaxError) as e:
            from tkinter import messagebox

            messagebox.showerror(
                "Erro de Entrada",
                f"Coordenadas inválidas. Por favor, verifique o formato.\n\nErro: {e}",
            )

    # ==============================
    # POPUP DE TRANSFORMAÇÕES
    # ==============================
    def abrir_transformacoes_popup(self):
        if not self.selected_obj_name:
            from tkinter import messagebox

            messagebox.showerror("Erro", "Selecione um objeto para transformar.")
            return

        self.lista_objetos.unbind("<<ListboxSelect>>")

        popup = tk.Toplevel(self)
        popup.title(f"Transformar {self.selected_obj_name}")
        popup.geometry("350x350")
        popup.protocol(
            "WM_DELETE_WINDOW", lambda: self.fechar_popup_transformacao(popup)
        )

        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)

        frame_t = ttk.Frame(notebook)
        notebook.add(frame_t, text="Translação")
        tk.Label(frame_t, text="dx:").grid(row=0, column=0, sticky="w")
        dx_t = tk.Entry(frame_t)
        dx_t.grid(row=0, column=1)
        tk.Label(frame_t, text="dy:").grid(row=1, column=0, sticky="w")
        dy_t = tk.Entry(frame_t)
        dy_t.grid(row=1, column=1)
        tk.Button(
            frame_t,
            text="Aplicar",
            command=lambda: self.aplicar_transformacao(
                "translacao",
                dx=float(dx_t.get()) if dx_t.get() else 0.0,
                dy=float(dy_t.get()) if dy_t.get() else 0.0,
                popup=popup,
            ),
        ).grid(row=2, columnspan=2, pady=10)

        frame_s = ttk.Frame(notebook)
        notebook.add(frame_s, text="Escalonamento")
        tk.Label(frame_s, text="sx:").grid(row=0, column=0, sticky="w")
        sx_s = tk.Entry(frame_s)
        sx_s.grid(row=0, column=1)
        tk.Label(frame_s, text="sy:").grid(row=1, column=0, sticky="w")
        sy_s = tk.Entry(frame_s)
        sy_s.grid(row=1, column=1)
        tk.Button(
            frame_s,
            text="Aplicar",
            command=lambda: self.aplicar_transformacao(
                "escalonamento_natural",
                sx=float(sx_s.get()) if sx_s.get() else 1.0,
                sy=float(sy_s.get()) if sy_s.get() else 1.0,
                popup=popup,
            ),
        ).grid(row=2, columnspan=2, pady=10)

        frame_r = ttk.Frame(notebook)
        notebook.add(frame_r, text="Rotação")
        rot_frame = tk.Frame(frame_r)
        rot_frame.grid(row=0, column=0)

        tk.Label(rot_frame, text="Ângulo (graus):").grid(row=0, column=0, sticky="w")
        angulo_r = tk.Entry(rot_frame)
        angulo_r.grid(row=0, column=1)

        rot_var = tk.StringVar(value="centro_objeto")
        tk.Radiobutton(
            frame_r,
            text="Em torno do centro do objeto",
            variable=rot_var,
            value="centro_objeto",
        ).grid(row=1, sticky="w")
        tk.Radiobutton(
            frame_r,
            text="Em torno do centro do mundo (0,0)",
            variable=rot_var,
            value="centro_mundo",
        ).grid(row=2, sticky="w")

        frame_arbitrario = tk.Frame(frame_r)
        frame_arbitrario.grid(row=3, sticky="w")
        tk.Radiobutton(
            frame_arbitrario,
            text="Em torno de um ponto:",
            variable=rot_var,
            value="ponto_arbitrario",
        ).pack(side="left")

        x_r = tk.Entry(frame_arbitrario, width=5)
        y_r = tk.Entry(frame_arbitrario, width=5)
        tk.Label(frame_arbitrario, text="x:").pack(side="left", padx=(10, 0))
        x_r.pack(side="left")
        tk.Label(frame_arbitrario, text="y:").pack(side="left", padx=(5, 0))
        y_r.pack(side="left")

        tk.Button(
            frame_r,
            text="Aplicar",
            command=lambda: self.aplicar_transformacao(
                rot_var.get(),
                angulo=float(angulo_r.get()) if angulo_r.get() else 0.0,
                px=float(x_r.get()) if x_r.get() else 0.0,
                py=float(y_r.get()) if y_r.get() else 0.0,
                popup=popup,
            ),
        ).grid(row=4, columnspan=2, pady=10)

    def fechar_popup_transformacao(self, popup):
        self.lista_objetos.bind("<<ListboxSelect>>", self.on_obj_select)
        popup.destroy()

    def aplicar_transformacao(self, tipo_transf, **kwargs):
        obj = self.displayfile.get_by_name(self.selected_obj_name)
        if not obj:
            return

        try:
            if tipo_transf == "translacao":
                matriz = self.transformacoes.get_matriz_translacao(
                    kwargs["dx"], kwargs["dy"]
                )
                self.transformacoes.aplicar_transformacao_generica(obj, matriz)
            elif tipo_transf == "escalonamento_natural":
                self.transformacoes.aplicar_escalonamento_natural(
                    obj, kwargs["sx"], kwargs["sy"]
                )
            elif tipo_transf == "centro_mundo":
                angulo_rad = np.deg2rad(kwargs["angulo"])
                matriz = self.transformacoes.get_matriz_rotacao(angulo_rad)
                self.transformacoes.aplicar_transformacao_generica(obj, matriz)
            elif tipo_transf == "centro_objeto":
                angulo_rad = np.deg2rad(kwargs["angulo"])
                self.transformacoes.aplicar_rotacao_centro_objeto(obj, angulo_rad)
            elif tipo_transf == "ponto_arbitrario":
                angulo_rad = np.deg2rad(kwargs["angulo"])
                self.transformacoes.aplicar_rotacao_ponto_arbitrario(
                    obj, angulo_rad, kwargs["px"], kwargs["py"]
                )

            self.viewport.desenhar(self.displayfile)
            if "popup" in kwargs:
                self.fechar_popup_transformacao(kwargs["popup"])

        except (ValueError, TypeError) as e:
            from tkinter import messagebox

            messagebox.showerror(
                "Erro de Entrada",
                f"Entrada inválida. Verifique os valores numéricos.\n\nErro: {e}",
            )

    def pan(self, dx, dy):
        theta = np.deg2rad(self.viewport.window_angle)
        c = np.cos(theta)
        s = np.sin(theta)
        world_dx = dx * c - dy * s
        world_dy = dx * s + dy * c

        self.viewport.wmin = (
            self.viewport.wmin[0] + world_dx,
            self.viewport.wmin[1] + world_dy,
        )
        self.viewport.wmax = (
            self.viewport.wmax[0] + world_dx,
            self.viewport.wmax[1] + world_dy,
        )
        self.viewport.desenhar(self.displayfile)

    def zoom(self, fator):
        cx = (self.viewport.wmin[0] + self.viewport.wmax[0]) / 2
        cy = (self.viewport.wmin[1] + self.viewport.wmax[1]) / 2
        largura = (self.viewport.wmax[0] - self.viewport.wmin[0]) * fator
        altura = (self.viewport.wmax[1] - self.viewport.wmin[1]) * fator
        self.viewport.wmin = (cx - largura / 2, cy - altura / 2)
        self.viewport.wmax = (cx + largura / 2, cy + altura / 2)
        self.viewport.desenhar(self.displayfile)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sistema Básico de CG 2D - clipping integrado")
    root.geometry("900x650")
    app = App(root)
    app.mainloop()
