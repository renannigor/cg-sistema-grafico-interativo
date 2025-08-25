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
    def __init__(self, canvas, wmin, wmax, vpmin, vpmax):
        self.canvas = canvas
        self.wmin = wmin
        self.wmax = wmax
        self.vpmin = vpmin
        self.vpmax = vpmax

    def transformar(self, xw, yw):
        """Transforma coordenadas do mundo (xw,yw) para viewport"""
        xvp = ((xw - self.wmin[0]) / (self.wmax[0] - self.wmin[0])) * (
            self.vpmax[0] - self.vpmin[0]
        ) + self.vpmin[0]
        yvp = (1 - (yw - self.wmin[1]) / (self.wmax[1] - self.wmin[1])) * (
            self.vpmax[1] - self.vpmin[1]
        ) + self.vpmin[1]
        return (xvp, yvp)

    def desenhar(self, displayfile: DisplayFile):
        self.canvas.delete("all")

        # === Desenhar eixos X e Y ===
        origem = self.transformar(0, 0)
        x_min = self.transformar(self.wmin[0], 0)
        x_max = self.transformar(self.wmax[0], 0)
        y_min = self.transformar(0, self.wmin[1])
        y_max = self.transformar(0, self.wmax[1])

        # Eixo X
        self.canvas.create_line(x_min[0], x_min[1], x_max[0], x_max[1], fill="gray", dash=(2, 2))
        # Eixo Y
        self.canvas.create_line(y_min[0], y_min[1], y_max[0], y_max[1], fill="gray", dash=(2, 2))

        # === Desenhar objetos ===
        for obj in displayfile.listar():
            coords_vp = [self.transformar(x, y) for (x, y) in obj.coords]
            if obj.tipo == TipoObjeto.PONTO.value:
                x, y = coords_vp[0]
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="black")
            elif obj.tipo == TipoObjeto.RETA.value:
                self.canvas.create_line(coords_vp, fill="blue")
            elif obj.tipo == TipoObjeto.WIREFRAME.value:
                self.canvas.create_polygon(coords_vp, outline="red", fill="", width=2)


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()

        # Layout principal: esquerda = canvas | direita = lista de objetos
        frame_main = tk.Frame(self)
        frame_main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(frame_main, width=400, height=400, bg="white")
        self.canvas.pack(side="left", padx=5, pady=5)

        # Lista de objetos
        self.lista_objetos = tk.Listbox(frame_main, width=30, height=25)
        self.lista_objetos.pack(side="right", padx=5, pady=5, fill="y")

        self.displayfile = DisplayFile()
        self.viewport = Viewport(
            self.canvas, wmin=(-100, -100), wmax=(100, 100), vpmin=(0, 0), vpmax=(400, 400)
        )

        # Botão para abrir popup
        tk.Button(self, text="Adicionar Objeto", command=self.abrir_popup).pack(pady=5)

        # Botões de navegação
        tk.Button(self, text="Esquerda", command=lambda: self.pan(-10, 0)).pack(side="left")
        tk.Button(self, text="Direita", command=lambda: self.pan(10, 0)).pack(side="left")
        tk.Button(self, text="Cima", command=lambda: self.pan(0, 10)).pack(side="left")
        tk.Button(self, text="Baixo", command=lambda: self.pan(0, -10)).pack(side="left")
        tk.Button(self, text="Zoom +", command=lambda: self.zoom(0.9)).pack(side="left")
        tk.Button(self, text="Zoom -", command=lambda: self.zoom(1.1)).pack(side="left")

        # desenhar eixos no início
        self.viewport.desenhar(self.displayfile)

    # ==============================
    # POPUP DE INSERÇÃO DE OBJETOS
    # ==============================
    def abrir_popup(self):
        popup = tk.Toplevel(self)
        popup.title("Incluir Objeto")
        popup.geometry("300x250")

        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both")

        # === Aba Ponto ===
        frame_ponto = ttk.Frame(notebook)
        notebook.add(frame_ponto, text="Ponto")
        nome_ponto = tk.Entry(frame_ponto)
        x_ponto = tk.Entry(frame_ponto)
        y_ponto = tk.Entry(frame_ponto)
        tk.Label(frame_ponto, text="Nome:").grid(row=0, column=0); nome_ponto.grid(row=0, column=1)
        tk.Label(frame_ponto, text="x:").grid(row=1, column=0); x_ponto.grid(row=1, column=1)
        tk.Label(frame_ponto, text="y:").grid(row=2, column=0); y_ponto.grid(row=2, column=1)
        tk.Button(
            frame_ponto, text="Adicionar",
            command=lambda: self.add_obj(
                nome_ponto.get(), TipoObjeto.PONTO.value,
                [(float(x_ponto.get()), float(y_ponto.get()))], popup)
        ).grid(row=3, columnspan=2)

        # === Aba Reta ===
        frame_reta = ttk.Frame(notebook)
        notebook.add(frame_reta, text="Reta")
        nome_reta = tk.Entry(frame_reta)
        x1 = tk.Entry(frame_reta); y1 = tk.Entry(frame_reta)
        x2 = tk.Entry(frame_reta); y2 = tk.Entry(frame_reta)
        tk.Label(frame_reta, text="Nome:").grid(row=0, column=0); nome_reta.grid(row=0, column=1)
        tk.Label(frame_reta, text="x1:").grid(row=1, column=0); x1.grid(row=1, column=1)
        tk.Label(frame_reta, text="y1:").grid(row=2, column=0); y1.grid(row=2, column=1)
        tk.Label(frame_reta, text="x2:").grid(row=3, column=0); x2.grid(row=3, column=1)
        tk.Label(frame_reta, text="y2:").grid(row=4, column=0); y2.grid(row=4, column=1)
        tk.Button(
            frame_reta, text="Adicionar",
            command=lambda: self.add_obj(
                nome_reta.get(), TipoObjeto.RETA.value,
                [(float(x1.get()), float(y1.get())), (float(x2.get()), float(y2.get()))], popup)
        ).grid(row=5, columnspan=2)

        # === Aba Wireframe ===
        frame_wire = ttk.Frame(notebook)
        notebook.add(frame_wire, text="Wireframe")
        nome_wire = tk.Entry(frame_wire)
        coords_entry = tk.Entry(frame_wire, width=25)
        tk.Label(frame_wire, text="Nome:").grid(row=0, column=0); nome_wire.grid(row=0, column=1)
        coords_entry.grid(row=1, column=0, columnspan=2, pady=5)
        tk.Label(frame_wire, text="Formato: (x1,y1),(x2,y2),...").grid(row=2, column=0, columnspan=2)
        tk.Button(
            frame_wire, text="Adicionar",
            command=lambda: self.add_obj(nome_wire.get(), TipoObjeto.WIREFRAME.value, list(eval(coords_entry.get())), popup)
        ).grid(row=3, columnspan=2)

    def add_obj(self, nome, tipo, coords, popup):
        obj = Objeto(nome if nome else tipo, tipo, coords)
        self.displayfile.adicionar(obj)
        self.viewport.desenhar(self.displayfile)
        self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
        popup.destroy()

    # ==============================
    # PAN & ZOOM
    # ==============================
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
    app = App(root)
    app.mainloop()
