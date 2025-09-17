import tkinter as tk
from tkinter import ttk, colorchooser, messagebox
from typing import List, Tuple
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
    POLIGONO = "poligono"


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
                p1 = verts[i]
                p2 = verts[(i + 1) % len(verts)]

                inside1 = esta_dentro(p1, aresta)
                inside2 = esta_dentro(p2, aresta)

                if inside1 and inside2:
                    novos_verts.append(p2)
                elif inside1 and not inside2:
                    novos_verts.append(intersecao(p1, p2, aresta))
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

        if not coords_clipadas:
            return None

        return Objeto(obj.nome, obj.tipo, coords_clipadas, obj.cor, obj.preenchido)


class Viewport:
    def __init__(self, canvas, wmin, wmax):
        self.canvas = canvas
        self.wmin = wmin
        self.wmax = wmax
        self.window_angle = 0.0
        self.clipping = Clipping()
        
    def set_window_angle(self, ang_deg):
        self.window_angle = ang_deg

    def _rotacionar_ponto_em_torno_de(self, x, y, cx, cy, ang_rad):
        tx = x - cx
        ty = y - cy
        c = np.cos(ang_rad)
        s = np.sin(ang_rad)
        xr = c * tx - s * ty
        yr = s * tx + c * ty
        return (xr + cx, yr + cy)

    def transformar(self, xw, yw, w_min, w_max, vpmin, vpmax):
        xw_range = w_max[0] - w_min[0]
        yw_range = w_max[1] - w_min[1]
        xvp_range = vpmax[0] - vpmin[0]
        yvp_range = vpmax[1] - vpmin[1]

        if xw_range == 0 or yw_range == 0: return (0, 0)

        xvp = vpmin[0] + ((xw - w_min[0]) / xw_range) * xvp_range
        yvp = vpmin[1] + (1 - (yw - w_min[1]) / yw_range) * yvp_range
        return (xvp, yvp)

    def desenhar(self, displayfile: DisplayFile, alg_reta_clip: str):
        self.canvas.delete("all")

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        canvas_margin = 10

        if canvas_width <= (2 * canvas_margin) or canvas_height <= (2 * canvas_margin): return

        available_width = canvas_width - (2 * canvas_margin)
        available_height = canvas_height - (2 * canvas_margin)

        w_width = self.wmax[0] - self.wmin[0]
        w_height = self.wmax[1] - self.wmin[1]
        
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
        
        cx = (self.wmin[0] + self.wmax[0]) / 2.0
        cy = (self.wmin[1] + self.wmax[1]) / 2.0
        ang_rad = -np.deg2rad(self.window_angle)

        def transform_coord(xw, yw):
            xr, yr = self._rotacionar_ponto_em_torno_de(xw, yw, cx, cy, ang_rad)
            return self.transformar(xr, yr, self.wmin, self.wmax, vpmin, vpmax)

        origem_vp = transform_coord(0, 0)
        p_inicio_eixo_x = transform_coord(self.wmin[0], 0)
        p_fim_eixo_x = transform_coord(self.wmax[0], 0)
        p_inicio_eixo_y = transform_coord(0, self.wmin[1])
        p_fim_eixo_y = transform_coord(0, self.wmax[1])
        
        self.canvas.create_line(p_inicio_eixo_x[0], origem_vp[1], p_fim_eixo_x[0], origem_vp[1], fill="gray", dash=(2, 2))
        self.canvas.create_line(origem_vp[0], p_inicio_eixo_y[1], origem_vp[0], p_fim_eixo_y[1], fill="gray", dash=(2, 2))

        for obj in displayfile.listar():
            obj_clipado = self.clipping.clip(obj, self.wmin, self.wmax, alg_reta_clip)

            if not obj_clipado:
                continue

            coords_vp = [transform_coord(x, y) for (x, y) in obj_clipado.coords]

            if obj_clipado.tipo == TipoObjeto.PONTO.value:
                x, y = coords_vp[0]
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=obj_clipado.cor, outline=obj_clipado.cor)
            elif obj_clipado.tipo == TipoObjeto.RETA.value:
                flat = [c for p in coords_vp for c in p]
                if len(flat) >= 4:
                    self.canvas.create_line(*flat, fill=obj_clipado.cor, width=2)
            elif obj_clipado.tipo == TipoObjeto.POLIGONO.value:
                if len(coords_vp) > 1:
                    if obj_clipado.preenchido:
                        self.canvas.create_polygon(coords_vp, fill=obj_clipado.cor, outline="")
                    else:
                        self.canvas.create_polygon(coords_vp, fill="", outline=obj_clipado.cor, width=2)


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        self.selected_obj_name = None
        self.transformacoes = Transformacoes()
        self.displayfile = DisplayFile()
        
        frame_main = tk.Frame(self)
        frame_main.pack(fill="both", expand=True)

        frame_main.grid_columnconfigure(0, weight=1)
        frame_main.grid_columnconfigure(1, weight=0)
        frame_main.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame_main, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.viewport = Viewport(self.canvas, wmin=(-100, -100), wmax=(100, 100))

        self.lista_objetos = tk.Listbox(frame_main, width=30)
        self.lista_objetos.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.lista_objetos.bind("<<ListboxSelect>>", self.on_obj_select)

        # --- PAINEL DE CONTROLES REESTRUTURADO ---
        main_controls_container = tk.Frame(self)
        main_controls_container.pack(fill="x", padx=5, pady=(0, 5))

        top_row_frame = tk.Frame(main_controls_container)
        top_row_frame.pack(fill="x")

        bottom_row_frame = tk.Frame(main_controls_container)
        bottom_row_frame.pack(fill="x", pady=(5,0))

        # --- Fileira Superior ---
        tk.Button(top_row_frame, text="Adicionar Objeto", command=self.abrir_popup_objetos).pack(side="left", padx=2)
        tk.Button(top_row_frame, text="Aplicar Transformação", command=self.abrir_transformacoes_popup).pack(side="left", padx=2)
        
        tk.Label(top_row_frame, text="  Navegação:").pack(side="left", padx=(10,0))
        tk.Button(top_row_frame, text="←", command=lambda: self.pan(10, 0)).pack(side="left")
        tk.Button(top_row_frame, text="→", command=lambda: self.pan(-10, 0)).pack(side="left")
        tk.Button(top_row_frame, text="↑", command=lambda: self.pan(0, -10)).pack(side="left")
        tk.Button(top_row_frame, text="↓", command=lambda: self.pan(0, 10)).pack(side="left")
        tk.Button(top_row_frame, text="Zoom +", command=lambda: self.zoom(0.9)).pack(side="left", padx=(5, 2))
        tk.Button(top_row_frame, text="Zoom -", command=lambda: self.zoom(1.1)).pack(side="left")
        tk.Button(top_row_frame, text="Rot. Win", command=self.popup_rotacionar_window).pack(side="left", padx=(5, 2))

        # --- Fileira Inferior ---
        clipping_frame = tk.Frame(bottom_row_frame, borderwidth=1, relief="groove")
        clipping_frame.pack() # Centraliza por padrão

        tk.Label(clipping_frame, text="Algoritmo de Clipping de Reta:").pack(side="left", padx=(5,0))
        self.alg_reta_clip_var = tk.StringVar(value="cs")
        tk.Radiobutton(clipping_frame, text="Cohen-Sutherland", variable=self.alg_reta_clip_var, value="cs", command=self.redesenhar).pack(side="left")
        tk.Radiobutton(clipping_frame, text="Liang-Barsky", variable=self.alg_reta_clip_var, value="lb", command=self.redesenhar).pack(side="left", padx=(0,5))

        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.selected_color = "black"

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

    def on_canvas_resize(self, event):
        self.redesenhar()

    def on_obj_select(self, event):
        selecao = self.lista_objetos.curselection()
        if selecao:
            entry = self.lista_objetos.get(selecao[0])
            self.selected_obj_name = entry.rsplit(" (", 1)[0] if " (" in entry else entry
        else:
            self.selected_obj_name = None

    def abrir_popup_objetos(self):
        popup = tk.Toplevel(self)
        popup.title("Incluir Objeto")
        popup.geometry("300x320")

        frame_cor = tk.Frame(popup, pady=5)
        frame_cor.pack(fill="x", padx=10)
        tk.Label(frame_cor, text="Cor:").pack(side="left")
        self.cor_preview = tk.Canvas(frame_cor, width=20, height=20, bg="black", relief="solid")
        self.cor_preview.pack(side="left", padx=5)
        tk.Button(frame_cor, text="Escolher Cor", command=self.escolher_cor).pack(side="left")

        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=5)

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

        frame_poli = ttk.Frame(notebook)
        notebook.add(frame_poli, text="Polígono")
        nome_poli = tk.Entry(frame_poli)
        coords_entry = tk.Entry(frame_poli, width=30)
        preenchido_var = tk.BooleanVar()
        tk.Label(frame_poli, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_poli.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_poli, text="Coordenadas:").grid(row=1, column=0, columnspan=2, sticky="w")
        coords_entry.grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")
        tk.Label(frame_poli, text="Formato: [(x1,y1),(x2,y2),...]").grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(frame_poli, text="Preenchido", variable=preenchido_var).grid(row=4, columnspan=2, pady=5)
        tk.Button(frame_poli, text="Adicionar", command=lambda: self.add_obj(nome_poli.get(), TipoObjeto.POLIGONO.value, list(ast.literal_eval(coords_entry.get())), popup, self.selected_color, preenchido_var.get())).grid(row=5, columnspan=2, pady=10)

    def escolher_cor(self):
        cor_codigo = colorchooser.askcolor(title="Escolha uma cor")[1]
        if cor_codigo:
            self.selected_color = cor_codigo
            self.cor_preview.config(bg=cor_codigo)

    def add_obj(self, nome, tipo, coords, popup, cor, preenchido=False):
        try:
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

    def fechar_popup_transformacao(self, popup):
        self.lista_objetos.bind("<<ListboxSelect>>", self.on_obj_select)
        popup.destroy()

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
    root.title("SGI 2D - Clipping (Tamanho Fixo)")
    root.geometry("1000x700")
    root.resizable(False, False)
    app = App(root)
    app.mainloop()