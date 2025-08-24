import tkinter as tk
from tkinter import ttk
from typing import List, Tuple
from enum import Enum


class TipoObjeto(Enum):
    PONTO = "ponto"
    RETA = "reta"
    WIREFRAME = "wireframe"


class Objeto:
    def __init__(self, nome: str, tipo: str, coords: List[Tuple[float, float]]):
        self.nome = nome
        self.tipo = tipo
        self.coords = coords


class DisplayFile:
    def __init__(self):
        self.objetos = []

    def adicionar(self, obj: Objeto):
        self.objetos.append(obj)

    def listar(self):
        return self.objetos


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
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="black")
                            
            elif obj.tipo == TipoObjeto.RETA.value:
                self.canvas.create_line(coords_vp, fill="blue", width=2)
                            
            elif obj.tipo == TipoObjeto.WIREFRAME.value:
                if len(coords_vp) > 1:
                    for i in range(len(coords_vp) - 1):
                        p1 = coords_vp[i]
                        p2 = coords_vp[i+1]
                        self.canvas.create_line(p1, p2, fill="red", width=2)
                                    
                    p_last = coords_vp[-1]
                    p_first = coords_vp[0]
                    self.canvas.create_line(p_last, p_first, fill="red", width=2)


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        frame_main = tk.Frame(self)
        frame_main.pack(fill="both", expand=True)
        
        frame_main.grid_columnconfigure(0, weight=1)
        frame_main.grid_columnconfigure(1, weight=0)

        self.canvas = tk.Canvas(frame_main, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.lista_objetos = tk.Listbox(frame_main, width=30)
        self.lista_objetos.grid(row=0, column=1, sticky="ns", padx=5, pady=5)

        control_frame = tk.Frame(self)
        control_frame.pack(fill="x", padx=5, pady=5)

        self.displayfile = DisplayFile()
        self.viewport = Viewport(
            self.canvas, wmin=(-100, -100), wmax=(100, 100)
        )

        tk.Button(control_frame, text="Adicionar Objeto", command=self.abrir_popup).pack(side="left", padx=2)

        tk.Button(control_frame, text="←", command=lambda: self.pan(-10, 0)).pack(side="left")
        tk.Button(control_frame, text="→", command=lambda: self.pan(10, 0)).pack(side="left")
        tk.Button(control_frame, text="↑", command=lambda: self.pan(0, 10)).pack(side="left")
        tk.Button(control_frame, text="↓", command=lambda: self.pan(0, -10)).pack(side="left")
        tk.Button(control_frame, text="Zoom +", command=lambda: self.zoom(0.9)).pack(side="left", padx=(10, 2))
        tk.Button(control_frame, text="Zoom -", command=lambda: self.zoom(1.1)).pack(side="left")

        self.canvas.bind("<Configure>", self.on_canvas_resize)
    
    def on_canvas_resize(self, event):
        self.viewport.desenhar(self.displayfile)


    # ==============================
    # POPUP DE INSERÇÃO DE OBJETOS
    # ==============================
    def abrir_popup(self):
        popup = tk.Toplevel(self)
        popup.title("Incluir Objeto")
        popup.geometry("300x250")

        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # === Aba Ponto ===
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
                [(float(x_ponto.get()), float(y_ponto.get()))], popup)
        ).grid(row=3, columnspan=2, pady=10)

        # === Aba Reta ===
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
                [(float(x1.get()), float(y1.get())), (float(x2.get()), float(y2.get()))], popup)
        ).grid(row=5, columnspan=2, pady=10)

        # === Aba Wireframe ===
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
            command=lambda: self.add_obj(nome_wire.get(), TipoObjeto.WIREFRAME.value, list(eval(coords_entry.get())), popup)
        ).grid(row=4, columnspan=2, pady=10)

    def add_obj(self, nome, tipo, coords, popup):
        try:
            coords_float = [(float(x), float(y)) for x, y in coords]
            obj = Objeto(nome if nome else tipo, tipo, coords_float)
            self.displayfile.adicionar(obj)
            self.viewport.desenhar(self.displayfile)
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
            popup.destroy()
        except (ValueError, TypeError, SyntaxError) as e:
            from tkinter import messagebox
            messagebox.showerror("Erro de Entrada", f"Coordenadas inválidas. Por favor, verifique o formato.\n\nErro: {e}")


    # ==============================
    # PAN & ZOOM
    # ==============================
    def pan(self, dx, dy):
        pan_step_x = (self.viewport.wmax[0] - self.viewport.wmin[0]) / 40
        pan_step_y = (self.viewport.wmax[1] - self.viewport.wmin[1]) / 40
        
        final_dx = (dx / 10 if dx != 0 else 0) * pan_step_x * 10
        final_dy = (dy / 10 if dy != 0 else 0) * pan_step_y * 10

        self.viewport.wmin = (self.viewport.wmin[0] + final_dx, self.viewport.wmin[1] + final_dy)
        self.viewport.wmax = (self.viewport.wmax[0] + final_dx, self.viewport.wmax[1] + final_dy)
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
