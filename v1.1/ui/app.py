import tkinter as tk
from tkinter import ttk

from models.tipo_objeto import TipoObjeto
from models.display_file import DisplayFile
from models.objeto import Objeto
from core.viewport import Viewport


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
        self.viewport = Viewport(self.canvas, wmin=(-100, -100), wmax=(100, 100))

        tk.Button(
            control_frame, text="Adicionar Objeto", command=self.abrir_popup
        ).pack(side="left", padx=2)

        tk.Button(control_frame, text="←", command=lambda: self.pan(-10, 0)).pack(
            side="left"
        )
        tk.Button(control_frame, text="→", command=lambda: self.pan(10, 0)).pack(
            side="left"
        )
        tk.Button(control_frame, text="↑", command=lambda: self.pan(0, 10)).pack(
            side="left"
        )
        tk.Button(control_frame, text="↓", command=lambda: self.pan(0, -10)).pack(
            side="left"
        )
        tk.Button(control_frame, text="Zoom +", command=lambda: self.zoom(0.9)).pack(
            side="left", padx=(10, 2)
        )
        tk.Button(control_frame, text="Zoom -", command=lambda: self.zoom(1.1)).pack(
            side="left"
        )

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
            ),
        ).grid(row=3, columnspan=2, pady=10)

        # === Aba Reta ===
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
            ),
        ).grid(row=5, columnspan=2, pady=10)

        # === Aba Wireframe ===
        frame_wire = ttk.Frame(notebook)
        notebook.add(frame_wire, text="Wireframe")
        nome_wire = tk.Entry(frame_wire)
        coords_entry = tk.Entry(frame_wire, width=30)
        tk.Label(frame_wire, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_wire.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_wire, text="Coordenadas:").grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        coords_entry.grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")
        tk.Label(frame_wire, text="Formato: (x1,y1),(x2,y2),...").grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        tk.Button(
            frame_wire,
            text="Adicionar",
            command=lambda: self.add_obj(
                nome_wire.get(),
                TipoObjeto.WIREFRAME.value,
                list(eval(coords_entry.get())),
                popup,
            ),
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

            messagebox.showerror(
                "Erro de Entrada",
                f"Coordenadas inválidas. Por favor, verifique o formato.\n\nErro: {e}",
            )

    # ==============================
    # PAN & ZOOM
    # ==============================
    def pan(self, dx, dy):
        pan_step_x = (self.viewport.wmax[0] - self.viewport.wmin[0]) / 40
        pan_step_y = (self.viewport.wmax[1] - self.viewport.wmin[1]) / 40

        final_dx = (dx / 10 if dx != 0 else 0) * pan_step_x * 10
        final_dy = (dy / 10 if dy != 0 else 0) * pan_step_y * 10

        self.viewport.wmin = (
            self.viewport.wmin[0] + final_dx,
            self.viewport.wmin[1] + final_dy,
        )
        self.viewport.wmax = (
            self.viewport.wmax[0] + final_dx,
            self.viewport.wmax[1] + final_dy,
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
