import tkinter as tk
from tkinter import ttk, colorchooser
from typing import List, Tuple
from enum import Enum
import numpy as np

class Transformacoes:
    @staticmethod
    def get_matriz_translacao(dx, dy):
        return np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])

    @staticmethod
    def get_matriz_escalonamento(sx, sy):
        return np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def get_matriz_rotacao(angulo_rad):
        c = np.cos(angulo_rad)
        s = np.sin(angulo_rad)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

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
    def __init__(self, nome: str, tipo: str, coords: List[Tuple[float, float]], cor: str = "black"):
        self.nome = nome
        self.tipo = tipo
        self.coords = coords
        self.cor = cor


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
            
            if (x_min - margem <= x <= x_max + margem and
                y_min - margem <= y <= y_max + margem):
                return obj
        return None


class Viewport:
    def __init__(self, canvas, wmin, wmax): 
        self.canvas = canvas
        self.wmin = wmin
        self.wmax = wmax

    def transformar(self, xw, yw, w_adj_min, w_adj_max, vpmin, vpmax): 
        """Transforma coordenadas do mundo (xw,yw) para viewport"""
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
        vpmin = (0, 0)
        vpmax = (vp_width, vp_height)

        if vp_width == 0 or vp_height == 0:
            return

        vp_aspect = vp_width / vp_height
        w_width = self.wmax[0] - self.wmin[0]
        w_height = self.wmax[1] - self.wmin[1]
        w_aspect = w_width / w_height
        
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
        
        def transform(xw, yw):
            return self.transformar(xw, yw, w_adj_min, w_adj_max, vpmin, vpmax)

        origem = transform(0, 0)
        x_axis_start = transform(w_adj_min[0], 0)
        x_axis_end = transform(w_adj_max[0], 0)
        y_axis_start = transform(0, w_adj_min[1])
        y_axis_end = transform(0, w_adj_max[1])
        
        self.canvas.create_line(x_axis_start[0], origem[1], x_axis_end[0], origem[1], fill="gray", dash=(2, 2))
        self.canvas.create_line(origem[0], y_axis_start[1], origem[0], y_axis_end[1], fill="gray", dash=(2, 2))

        for obj in displayfile.listar():
            coords_vp = [transform(x, y) for (x, y) in obj.coords]
                        
            if obj.tipo == TipoObjeto.PONTO.value:
                x, y = coords_vp[0]
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=obj.cor)
                        
            elif obj.tipo == TipoObjeto.RETA.value:
                self.canvas.create_line(coords_vp, fill=obj.cor, width=2)
                        
            elif obj.tipo == TipoObjeto.WIREFRAME.value:
                if len(coords_vp) > 1:
                    for i in range(len(coords_vp) - 1):
                        p1 = coords_vp[i]
                        p2 = coords_vp[i+1]
                        self.canvas.create_line(p1, p2, fill=obj.cor, width=2)
                                        
                    p_last = coords_vp[-1]
                    p_first = coords_vp[0]
                    self.canvas.create_line(p_last, p_first, fill=obj.cor, width=2)


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
        self.viewport = Viewport(
            self.canvas, wmin=(-100, -100), wmax=(100, 100)
        )

        tk.Button(control_frame, text="Adicionar Objeto", command=self.abrir_popup).pack(side="left", padx=2)
        tk.Button(control_frame, text="Aplicar Transformação", command=self.abrir_transformacoes_popup).pack(side="left", padx=2)

        tk.Button(control_frame, text="←", command=lambda: self.pan(10, 0)).pack(side="left")
        tk.Button(control_frame, text="→", command=lambda: self.pan(-10, 0)).pack(side="left")
        tk.Button(control_frame, text="↑", command=lambda: self.pan(0, -10)).pack(side="left")
        tk.Button(control_frame, text="↓", command=lambda: self.pan(0, 10)).pack(side="left")
        tk.Button(control_frame, text="Zoom +", command=lambda: self.zoom(0.9)).pack(side="left", padx=(10, 2))
        tk.Button(control_frame, text="Zoom -", command=lambda: self.zoom(1.1)).pack(side="left")

        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.selected_color = "black"

    def on_canvas_resize(self, event):
        self.viewport.desenhar(self.displayfile)
        
    def on_obj_select(self, event):
        selecao = self.lista_objetos.curselection()
        if selecao:
            self.selected_obj_name = self.lista_objetos.get(selecao[0]).split(" ")[0]
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
        menu.add_command(label="Transformar Objeto", command=self.abrir_transformacoes_popup)
        menu.post(self.lista_objetos.winfo_rootx() + x, self.lista_objetos.winfo_rooty() + y)


    # ==============================
    # POPUP DE INSERÇÃO DE OBJETOS
    # ==============================
    def abrir_popup(self):
        popup = tk.Toplevel(self)
        popup.title("Incluir Objeto")
        popup.geometry("300x290")
        
        frame_cor = tk.Frame(popup, pady=5)
        frame_cor.pack(fill="x", padx=10)
        tk.Label(frame_cor, text="Cor do Objeto:").pack(side="left")
        
        color_button = tk.Button(frame_cor, text="Escolher Cor", command=self.escolher_cor)
        color_button.pack(side="left", padx=5)

        self.cor_preview = tk.Canvas(frame_cor, width=20, height=20, bg="black", borderwidth=1, relief="solid")
        self.cor_preview.pack(side="left")

        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=5)

        frame_ponto = ttk.Frame(notebook)
        notebook.add(frame_ponto, text="Ponto")
        nome_ponto = tk.Entry(frame_ponto)
        x_ponto = tk.Entry(frame_ponto)
        y_ponto = tk.Entry(frame_ponto)
        tk.Label(frame_ponto, text="Nome:").grid(row=0, column=0, sticky="w", pady=2); nome_ponto.grid(row=0, column=1)
        tk.Label(frame_ponto, text="x:").grid(row=1, column=0, sticky="w", pady=2); x_ponto.grid(row=1, column=1)
        tk.Label(frame_ponto, text="y:").grid(row=2, column=0, sticky="w", pady=2); y_ponto.grid(row=2, column=1)
        tk.Button(
            frame_ponto, text="Adicionar",
            command=lambda: self.add_obj(
                nome_ponto.get(), TipoObjeto.PONTO.value,
                [(float(x_ponto.get()), float(y_ponto.get()))], popup, self.selected_color)
        ).grid(row=3, columnspan=2, pady=10)

        frame_reta = ttk.Frame(notebook)
        notebook.add(frame_reta, text="Reta")
        nome_reta = tk.Entry(frame_reta)
        x1 = tk.Entry(frame_reta); y1 = tk.Entry(frame_reta)
        x2 = tk.Entry(frame_reta); y2 = tk.Entry(frame_reta)
        tk.Label(frame_reta, text="Nome:").grid(row=0, column=0, sticky="w", pady=2); nome_reta.grid(row=0, column=1)
        tk.Label(frame_reta, text="x1:").grid(row=1, column=0, sticky="w", pady=2); x1.grid(row=1, column=1)
        tk.Label(frame_reta, text="y1:").grid(row=2, column=0, sticky="w", pady=2); y1.grid(row=2, column=1)
        tk.Label(frame_reta, text="x2:").grid(row=3, column=0, sticky="w", pady=2); x2.grid(row=3, column=1)
        tk.Label(frame_reta, text="y2:").grid(row=4, column=0, sticky="w", pady=2); y2.grid(row=4, column=1)
        tk.Button(
            frame_reta, text="Adicionar",
            command=lambda: self.add_obj(
                nome_reta.get(), TipoObjeto.RETA.value,
                [(float(x1.get()), float(y1.get())), (float(x2.get()), float(y2.get()))], popup, self.selected_color)
        ).grid(row=5, columnspan=2, pady=10)

        frame_wire = ttk.Frame(notebook)
        notebook.add(frame_wire, text="Wireframe")
        nome_wire = tk.Entry(frame_wire)
        coords_entry = tk.Entry(frame_wire, width=30)
        tk.Label(frame_wire, text="Nome:").grid(row=0, column=0, sticky="w", pady=2); nome_wire.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_wire, text="Coordenadas:").grid(row=1, column=0, columnspan=2, sticky="w")
        coords_entry.grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")
        tk.Label(frame_wire, text="Formato: (x1,y1),(x2,y2),...").grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Button(
            frame_wire, text="Adicionar",
            command=lambda: self.add_obj(nome_wire.get(), TipoObjeto.WIREFRAME.value, list(eval(coords_entry.get())), popup, self.selected_color)
        ).grid(row=4, columnspan=2, pady=10)
    
    def escolher_cor(self):
        cor_codigo = colorchooser.askcolor(title="Escolha uma cor")[1]
        if cor_codigo:
            self.selected_color = cor_codigo
            self.cor_preview.config(bg=cor_codigo)

    def add_obj(self, nome, tipo, coords, popup, cor):
        try:
            coords_float = [(float(x), float(y)) for x, y in coords]
            obj = Objeto(nome if nome else tipo, tipo, coords_float, cor)
            self.displayfile.adicionar(obj)
            self.viewport.desenhar(self.displayfile)
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
            popup.destroy()
        except (ValueError, TypeError, SyntaxError) as e:
            from tkinter import messagebox
            messagebox.showerror("Erro de Entrada", f"Coordenadas inválidas. Por favor, verifique o formato.\n\nErro: {e}")

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
        popup.protocol("WM_DELETE_WINDOW", lambda: self.fechar_popup_transformacao(popup))

        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        frame_t = ttk.Frame(notebook)
        notebook.add(frame_t, text="Translação")
        tk.Label(frame_t, text="dx:").grid(row=0, column=0, sticky="w"); dx_t = tk.Entry(frame_t); dx_t.grid(row=0, column=1)
        tk.Label(frame_t, text="dy:").grid(row=1, column=0, sticky="w"); dy_t = tk.Entry(frame_t); dy_t.grid(row=1, column=1)
        tk.Button(frame_t, text="Aplicar", command=lambda: self.aplicar_transformacao(
            "translacao", 
            dx=float(dx_t.get()) if dx_t.get() else 0.0, 
            dy=float(dy_t.get()) if dy_t.get() else 0.0, 
            popup=popup)
        ).grid(row=2, columnspan=2, pady=10)
        
        frame_s = ttk.Frame(notebook)
        notebook.add(frame_s, text="Escalonamento")
        tk.Label(frame_s, text="sx:").grid(row=0, column=0, sticky="w"); sx_s = tk.Entry(frame_s); sx_s.grid(row=0, column=1)
        tk.Label(frame_s, text="sy:").grid(row=1, column=0, sticky="w"); sy_s = tk.Entry(frame_s); sy_s.grid(row=1, column=1)
        tk.Button(frame_s, text="Aplicar", command=lambda: self.aplicar_transformacao(
            "escalonamento_natural", 
            sx=float(sx_s.get()) if sx_s.get() else 1.0, 
            sy=float(sy_s.get()) if sy_s.get() else 1.0, 
            popup=popup)
        ).grid(row=2, columnspan=2, pady=10)
        
        frame_r = ttk.Frame(notebook)
        notebook.add(frame_r, text="Rotação")
        rot_frame = tk.Frame(frame_r)
        rot_frame.grid(row=0, column=0)
        
        tk.Label(rot_frame, text="Ângulo (graus):").grid(row=0, column=0, sticky="w")
        angulo_r = tk.Entry(rot_frame); angulo_r.grid(row=0, column=1)
        
        rot_var = tk.StringVar(value="centro_objeto")
        tk.Radiobutton(frame_r, text="Em torno do centro do objeto", variable=rot_var, value="centro_objeto").grid(row=1, sticky="w")
        tk.Radiobutton(frame_r, text="Em torno do centro do mundo (0,0)", variable=rot_var, value="centro_mundo").grid(row=2, sticky="w")
        
        frame_arbitrario = tk.Frame(frame_r)
        frame_arbitrario.grid(row=3, sticky="w")
        tk.Radiobutton(frame_arbitrario, text="Em torno de um ponto:", variable=rot_var, value="ponto_arbitrario").pack(side="left")
        
        x_r = tk.Entry(frame_arbitrario, width=5)
        y_r = tk.Entry(frame_arbitrario, width=5)
        tk.Label(frame_arbitrario, text="x:").pack(side="left", padx=(10,0))
        x_r.pack(side="left")
        tk.Label(frame_arbitrario, text="y:").pack(side="left", padx=(5,0))
        y_r.pack(side="left")
        
        tk.Button(frame_r, text="Aplicar", command=lambda: self.aplicar_transformacao(
            rot_var.get(), 
            angulo=float(angulo_r.get()) if angulo_r.get() else 0.0,
            px=float(x_r.get()) if x_r.get() else 0.0,
            py=float(y_r.get()) if y_r.get() else 0.0,
            popup=popup)
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
                matriz = self.transformacoes.get_matriz_translacao(kwargs['dx'], kwargs['dy'])
                self.transformacoes.aplicar_transformacao_generica(obj, matriz)
            elif tipo_transf == "escalonamento_natural":
                self.transformacoes.aplicar_escalonamento_natural(obj, kwargs['sx'], kwargs['sy'])
            elif tipo_transf == "centro_mundo":
                angulo_rad = np.deg2rad(kwargs['angulo'])
                matriz = self.transformacoes.get_matriz_rotacao(angulo_rad)
                self.transformacoes.aplicar_transformacao_generica(obj, matriz)
            elif tipo_transf == "centro_objeto":
                angulo_rad = np.deg2rad(kwargs['angulo'])
                self.transformacoes.aplicar_rotacao_centro_objeto(obj, angulo_rad)
            elif tipo_transf == "ponto_arbitrario":
                angulo_rad = np.deg2rad(kwargs['angulo'])
                self.transformacoes.aplicar_rotacao_ponto_arbitrario(obj, angulo_rad, kwargs['px'], kwargs['py'])
            
            self.viewport.desenhar(self.displayfile)
            if 'popup' in kwargs:
                self.fechar_popup_transformacao(kwargs['popup'])
                
        except (ValueError, TypeError) as e:
            from tkinter import messagebox
            messagebox.showerror("Erro de Entrada", f"Entrada inválida. Verifique os valores numéricos.\n\nErro: {e}")

    def pan(self, dx, dy):
        self.viewport.wmin = (self.viewport.wmin[0] + dx, self.viewport.wmin[1] + dy)
        self.viewport.wmax = (self.viewport.wmax[0] + dx, self.viewport.wmax[1] + dy)
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
    root.title("Sistema Básico de CG 2D")
    root.geometry("800x600")
    app = App(root)
    app.mainloop()