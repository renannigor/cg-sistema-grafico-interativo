# main.py
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox, simpledialog, filedialog
import numpy as np
import ast

# Importações dos nossos módulos
from geometria_2d import Transformacoes
from geometria_3d import Ponto3D, Transformacoes3D
from objetos import Objeto, Objeto3D, SuperficieBezier, SuperficieBSpline, TipoObjeto
from renderizacao import Viewport, Camera, DisplayFile


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        self.selected_obj_name = None
        self.transformacoes = Transformacoes()
        self.displayfile = DisplayFile()

        self.camera = Camera(
            vrp=Ponto3D(0, 50, 200), p_foco=Ponto3D(0, 0, 0), vup=(0, 1, 0), d=300
        )

        frame_main = tk.Frame(self)
        frame_main.pack(fill="both", expand=True)
        frame_main.grid_columnconfigure(0, weight=1)
        frame_main.grid_columnconfigure(1, weight=0)
        frame_main.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame_main, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.viewport = Viewport(
            self.canvas, wmin=(-100, -100), wmax=(100, 100), camera=self.camera
        )

        self.lista_objetos = tk.Listbox(frame_main, width=30)
        self.lista_objetos.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.lista_objetos.bind("<<ListboxSelect>>", self.on_obj_select)

        main_controls_container = tk.Frame(self)
        main_controls_container.pack(fill="x", padx=5, pady=(0, 5))
        top_row = tk.Frame(main_controls_container)
        top_row.pack(fill="x")
        mid_row = tk.Frame(main_controls_container)
        mid_row.pack(fill="x", pady=2)
        proj_row = tk.Frame(main_controls_container)
        proj_row.pack(fill="x", pady=2)
        bottom_row = tk.Frame(main_controls_container)
        bottom_row.pack(fill="x")

        tk.Button(
            top_row, text="Adicionar Objeto", command=self.abrir_popup_objetos
        ).pack(side="left", padx=2)
        tk.Button(
            top_row, text="Carregar .obj", command=self.carregar_arquivo_obj
        ).pack(side="left", padx=2)
        tk.Button(
            top_row,
            text="Aplicar Transformação",
            command=self.abrir_transformacoes_popup,
        ).pack(side="left", padx=2)

        tk.Label(mid_row, text="Nav. 2D:").pack(side="left", padx=(10, 0))
        tk.Button(mid_row, text="←", command=lambda: self.pan(10, 0)).pack(side="left")
        tk.Button(mid_row, text="→", command=lambda: self.pan(-10, 0)).pack(side="left")
        tk.Button(mid_row, text="↑", command=lambda: self.pan(0, -10)).pack(side="left")
        tk.Button(mid_row, text="↓", command=lambda: self.pan(0, 10)).pack(side="left")
        tk.Button(mid_row, text="Zoom +", command=lambda: self.zoom(0.9)).pack(
            side="left", padx=(5, 2)
        )
        tk.Button(mid_row, text="Zoom -", command=lambda: self.zoom(1.1)).pack(
            side="left"
        )
        tk.Button(mid_row, text="Rot. Win", command=self.popup_rotacionar_window).pack(
            side="left", padx=(5, 2)
        )

        tk.Label(proj_row, text="Nav. 3D (Câmera):").pack(side="left")
        tk.Button(
            proj_row,
            text="Frente",
            command=lambda: (self.camera.mover_relativo(0, 0, 10), self.redesenhar()),
        ).pack(side="left")
        tk.Button(
            proj_row,
            text="Trás",
            command=lambda: (self.camera.mover_relativo(0, 0, -10), self.redesenhar()),
        ).pack(side="left")
        tk.Button(
            proj_row,
            text="Esq.",
            command=lambda: (self.camera.mover_relativo(10, 0, 0), self.redesenhar()),
        ).pack(side="left")
        tk.Button(
            proj_row,
            text="Dir.",
            command=lambda: (self.camera.mover_relativo(-10, 0, 0), self.redesenhar()),
        ).pack(side="left")
        tk.Button(
            proj_row,
            text="Cima",
            command=lambda: (self.camera.mover_relativo(0, 10, 0), self.redesenhar()),
        ).pack(side="left")
        tk.Button(
            proj_row,
            text="Baixo",
            command=lambda: (self.camera.mover_relativo(0, -10, 0), self.redesenhar()),
        ).pack(side="left")
        tk.Button(
            proj_row,
            text="⟲(Y)",
            command=lambda: (self.camera.girar(np.deg2rad(-5), 0), self.redesenhar()),
        ).pack(side="left", padx=(5, 0))
        tk.Button(
            proj_row,
            text="⟳(Y)",
            command=lambda: (self.camera.girar(np.deg2rad(5), 0), self.redesenhar()),
        ).pack(side="left")
        tk.Button(
            proj_row,
            text="⬆(P)",
            command=lambda: (self.camera.girar(0, np.deg2rad(-5)), self.redesenhar()),
        ).pack(side="left")
        tk.Button(
            proj_row,
            text="⬇(P)",
            command=lambda: (self.camera.girar(0, np.deg2rad(5)), self.redesenhar()),
        ).pack(side="left")

        self.projection_type_var = tk.StringVar(value="ortogonal")
        tk.Label(bottom_row, text="Projeção:").pack(side="left")
        tk.Radiobutton(
            bottom_row,
            text="Ortogonal",
            variable=self.projection_type_var,
            value="ortogonal",
            command=self.redesenhar,
        ).pack(side="left")
        tk.Radiobutton(
            bottom_row,
            text="Perspectiva",
            variable=self.projection_type_var,
            value="perspectiva",
            command=self.redesenhar,
        ).pack(side="left")

        tk.Label(bottom_row, text="Dist. COP (d):").pack(side="left", padx=(10, 0))
        self.dist_slider = tk.Scale(
            bottom_row,
            from_=50,
            to=1000,
            orient="horizontal",
            length=200,
            command=self.atualizar_distancia_projecao,
        )
        self.dist_slider.set(self.camera.d)
        self.dist_slider.pack(side="left")

        clipping_frame = tk.Frame(bottom_row, borderwidth=1, relief="groove")
        clipping_frame.pack(side="left", padx=10)
        tk.Label(clipping_frame, text="Clipping de Reta:").pack(side="left", padx=5)
        self.alg_reta_clip_var = tk.StringVar(value="cs")
        tk.Radiobutton(
            clipping_frame,
            text="Cohen-Sutherland",
            variable=self.alg_reta_clip_var,
            value="cs",
            command=self.redesenhar,
        ).pack(side="left")
        tk.Radiobutton(
            clipping_frame,
            text="Liang-Barsky",
            variable=self.alg_reta_clip_var,
            value="lb",
            command=self.redesenhar,
        ).pack(side="left", padx=(0, 5))

        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.selected_color = "black"

    def atualizar_distancia_projecao(self, valor_str):
        self.camera.set_distancia_plano_projecao(float(valor_str))
        self.redesenhar()

    def redesenhar(self):
        self.viewport.desenhar(
            self.displayfile,
            self.alg_reta_clip_var.get(),
            self.projection_type_var.get(),
        )

    def carregar_arquivo_obj(self):
        filepath = filedialog.askopenfilename(
            title="Abrir arquivo OBJ",
            filetypes=(("OBJ Files", "*.obj"), ("All files", "*.*")),
        )
        if not filepath:
            return

        vertices = []
        faces = []

        try:
            with open(filepath, "r") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == "v":
                        x, y, z = map(float, parts[1:4])
                        vertices.append(Ponto3D(x * 50, y * 50, z * 50))
                    elif parts[0] == "f":
                        indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                        faces.append(indices)

            # Checar se o arquivo deve ser lido como superfície de Bézier
            if not faces and len(vertices) > 0 and len(vertices) % 16 == 0:
                resp = messagebox.askyesno(
                    "Superfície de Bézier Detectada",
                    "O arquivo não contém faces e o número de vértices é um múltiplo de 16.\n"
                    "Deseja carregar como uma ou mais superfícies de Bézier?",
                )
                if resp:
                    num_superficies = len(vertices) // 16
                    for i in range(num_superficies):
                        pontos_controle = vertices[i * 16 : (i + 1) * 16]
                        nome_superficie = f"superficie_{filepath.split('/')[-1]}_{i}"
                        obj = SuperficieBezier(
                            nome_superficie, pontos_controle, self.selected_color
                        )
                        self.displayfile.adicionar(obj)
                        self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
                    self.redesenhar()
                    return

            arestas = set()
            for face in faces:
                for i in range(len(face)):
                    idx1 = face[i]
                    idx2 = face[(i + 1) % len(face)]
                    aresta = tuple(sorted((idx1, idx2)))
                    arestas.add(aresta)

            nome_obj = f"obj_{filepath.split('/')[-1]}"
            obj = Objeto3D(nome_obj, vertices, list(arestas), self.selected_color)
            self.displayfile.adicionar(obj)
            self.redesenhar()
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")

        except Exception as e:
            messagebox.showerror(
                "Erro ao Carregar OBJ", f"Não foi possível ler o arquivo.\n\nErro: {e}"
            )

    def popup_rotacionar_window(self):
        popup = tk.Toplevel(self)
        popup.title("Rotacionar Window")
        tk.Label(popup, text="Ângulo (graus, anti-horário):").pack(padx=8, pady=(8, 0))
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
            self.selected_obj_name = (
                entry.rsplit(" (", 1)[0] if " (" in entry else entry
            )
        else:
            self.selected_obj_name = None

    def abrir_popup_objetos(self):
        popup = tk.Toplevel(self)
        popup.title("Incluir Objeto")
        popup.geometry("450x500")

        frame_cor = tk.Frame(popup, pady=5)
        frame_cor.pack(fill="x", padx=10)
        tk.Label(frame_cor, text="Cor:").pack(side="left")
        self.cor_preview = tk.Canvas(
            frame_cor, width=20, height=20, bg="black", relief="solid"
        )
        self.cor_preview.pack(side="left", padx=5)
        tk.Button(frame_cor, text="Escolher Cor", command=self.escolher_cor).pack(
            side="left"
        )

        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=5)

        frame_ponto = ttk.Frame(notebook)
        notebook.add(frame_ponto, text="Ponto")
        nome_ponto, x_ponto, y_ponto = (
            tk.Entry(frame_ponto),
            tk.Entry(frame_ponto),
            tk.Entry(frame_ponto),
        )
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
            ),
        ).grid(row=3, columnspan=2, pady=10)

        frame_reta = ttk.Frame(notebook)
        notebook.add(frame_reta, text="Reta")
        nome_reta, x1, y1, x2, y2 = (
            tk.Entry(frame_reta),
            tk.Entry(frame_reta),
            tk.Entry(frame_reta),
            tk.Entry(frame_reta),
            tk.Entry(frame_reta),
        )
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
            ),
        ).grid(row=5, columnspan=2, pady=10)

        frame_poli = ttk.Frame(notebook)
        notebook.add(frame_poli, text="Polígono")
        nome_poli = tk.Entry(frame_poli)
        coords_entry_poli = tk.Entry(frame_poli, width=30)
        preenchido_var = tk.BooleanVar()
        tk.Label(frame_poli, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_poli.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_poli, text="Coordenadas:").grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        coords_entry_poli.grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")
        tk.Label(frame_poli, text="Formato: (x1,y1),(x2,y2),...").grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        tk.Checkbutton(frame_poli, text="Preenchido", variable=preenchido_var).grid(
            row=4, columnspan=2, pady=5
        )
        tk.Button(
            frame_poli,
            text="Adicionar",
            command=lambda: self.add_obj(
                nome_poli.get(),
                TipoObjeto.POLIGONO.value,
                list(ast.literal_eval(coords_entry_poli.get())),
                popup,
                self.selected_color,
                preenchido_var.get(),
            ),
        ).grid(row=5, columnspan=2, pady=10)

        frame_curva_bezier = ttk.Frame(notebook)
        notebook.add(frame_curva_bezier, text="Curva Bézier")
        nome_curva_bezier = tk.Entry(frame_curva_bezier)
        coords_entry_curva_bezier = tk.Entry(frame_curva_bezier, width=30)
        tk.Label(frame_curva_bezier, text="Nome:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        nome_curva_bezier.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_curva_bezier, text="Pontos de Controle:").grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        coords_entry_curva_bezier.grid(
            row=2, column=0, columnspan=2, pady=2, sticky="ew"
        )
        tk.Label(frame_curva_bezier, text="Formato: (x1,y1),(x2,y2),...").grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        tk.Label(frame_curva_bezier, text="Nº de pontos: 4, 7, 10, ...").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=4
        )
        tk.Button(
            frame_curva_bezier,
            text="Adicionar",
            command=lambda: self.add_obj(
                nome_curva_bezier.get(),
                TipoObjeto.CURVA.value,
                list(ast.literal_eval(coords_entry_curva_bezier.get())),
                popup,
                self.selected_color,
            ),
        ).grid(row=5, columnspan=2, pady=10)

        frame_bspline = ttk.Frame(notebook)
        notebook.add(frame_bspline, text="B-Spline")
        nome_bspline = tk.Entry(frame_bspline)
        coords_entry_bspline = tk.Entry(frame_bspline, width=30)
        tk.Label(frame_bspline, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_bspline.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_bspline, text="Pontos de Controle:").grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        coords_entry_bspline.grid(row=2, column=0, columnspan=2, pady=2, sticky="ew")
        tk.Label(frame_bspline, text="Formato: (x1,y1),(x2,y2),...").grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        tk.Label(frame_bspline, text="Mínimo de 4 pontos.").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=4
        )
        tk.Button(
            frame_bspline,
            text="Adicionar",
            command=lambda: self.add_obj(
                nome_bspline.get(),
                TipoObjeto.B_SPLINE.value,
                list(ast.literal_eval(coords_entry_bspline.get())),
                popup,
                self.selected_color,
            ),
        ).grid(row=5, columnspan=2, pady=10)

        frame_obj3d = ttk.Frame(notebook)
        notebook.add(frame_obj3d, text="Objeto 3D")
        nome_obj3d = tk.Entry(frame_obj3d)
        tk.Label(frame_obj3d, text="Nome:").grid(row=0, column=0, sticky="w", pady=2)
        nome_obj3d.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_obj3d, text="Vértices (x,y,z):").grid(
            row=1, column=0, sticky="w"
        )
        vertices_text = tk.Text(frame_obj3d, height=6)
        vertices_text.grid(row=2, column=0, columnspan=2, sticky="ew")
        tk.Label(frame_obj3d, text="Formato: (x1,y1,z1); (x2,y2,z2); ...").grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        tk.Label(frame_obj3d, text="Arestas (índices):").grid(
            row=4, column=0, sticky="w", pady=(5, 0)
        )
        arestas_text = tk.Text(frame_obj3d, height=6)
        arestas_text.grid(row=5, column=0, columnspan=2, sticky="ew")
        tk.Label(frame_obj3d, text="Formato: (0,1); (1,2); ...").grid(
            row=6, column=0, columnspan=2, sticky="w"
        )
        vertices_text.insert(
            "1.0",
            "(-50,-50,-50); (50,-50,-50); (50,50,-50); (-50,50,-50); (-50,-50,50); (50,-50,50); (50,50,50); (-50,50,50)",
        )
        arestas_text.insert(
            "1.0",
            "(0,1);(1,2);(2,3);(3,0);(4,5);(5,6);(6,7);(7,4);(0,4);(1,5);(2,6);(3,7)",
        )
        tk.Button(
            frame_obj3d,
            text="Adicionar",
            command=lambda: self.add_obj_3d(
                nome_obj3d.get(),
                vertices_text.get("1.0", "end-1c"),
                arestas_text.get("1.0", "end-1c"),
                popup,
                self.selected_color,
            ),
        ).grid(row=7, columnspan=2, pady=10)

        frame_superficie = ttk.Frame(notebook)
        notebook.add(frame_superficie, text="Superfície Bézier")
        nome_superficie = tk.Entry(frame_superficie)
        tk.Label(frame_superficie, text="Nome:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        nome_superficie.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_superficie, text="16 Pontos de Controle (4x4):").grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        pontos_text = tk.Text(frame_superficie, height=8)
        pontos_text.grid(row=2, column=0, columnspan=2, sticky="ew")
        tk.Label(
            frame_superficie, text="Formato: (x,y,z),(x,y,z),... ; (linha 2) ; ..."
        ).grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Button(
            frame_superficie,
            text="Adicionar",
            command=lambda: self.add_superficie(
                nome_superficie.get(),
                pontos_text.get("1.0", "end-1c"),
                popup,
                self.selected_color,
            ),
        ).grid(row=4, columnspan=2, pady=10)

        frame_superficie_bspline = ttk.Frame(notebook)
        notebook.add(frame_superficie_bspline, text="Superfície B-Spline")
        nome_superficie_bspline = tk.Entry(frame_superficie_bspline)
        tk.Label(frame_superficie_bspline, text="Nome:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        nome_superficie_bspline.grid(row=0, column=1, sticky="ew")
        tk.Label(frame_superficie_bspline, text="Pontos de Controle (N x M):").grid(
            row=1, column=0, columnspan=2, sticky="w"
        )
        pontos_text_bspline = tk.Text(frame_superficie_bspline, height=8)
        pontos_text_bspline.grid(row=2, column=0, columnspan=2, sticky="ew")
        tk.Label(
            frame_superficie_bspline,
            text="Formato: (x,y,z),(x,y,z),... ; (linha 2) ; ... (mín 4x4)",
        ).grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Button(
            frame_superficie_bspline,
            text="Adicionar",
            command=lambda: self.add_superficie_bspline(
                nome_superficie_bspline.get(),
                pontos_text_bspline.get("1.0", "end-1c"),
                popup,
                self.selected_color,
            ),
        ).grid(row=4, columnspan=2, pady=10)

    def escolher_cor(self):
        cor_codigo = colorchooser.askcolor(title="Escolha uma cor")[1]
        if cor_codigo:
            self.selected_color = cor_codigo
            self.cor_preview.config(bg=cor_codigo)

    def add_superficie(self, nome, pontos_str, popup, cor):
        try:
            linhas = [linha.strip() for linha in pontos_str.split(";") if linha.strip()]
            if len(linhas) != 4:
                raise ValueError(
                    "São necessárias 4 linhas de pontos de controle separadas por ';'."
                )

            pontos_controle_raw = []
            for linha_str in linhas:
                linha_pontos = ast.literal_eval(f"[{linha_str}]")
                if len(linha_pontos) != 4:
                    raise ValueError("Cada linha deve conter 4 pontos de controle.")
                pontos_controle_raw.extend(linha_pontos)

            if len(pontos_controle_raw) != 16:
                raise ValueError("É necessário um total de 16 pontos de controle.")

            pontos_controle = [Ponto3D(x, y, z) for x, y, z in pontos_controle_raw]

            obj = SuperficieBezier(
                nome if nome else f"superficie_{len(self.displayfile.objetos)}",
                pontos_controle,
                cor,
            )
            self.displayfile.adicionar(obj)
            self.redesenhar()
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
            popup.destroy()
        except Exception as e:
            messagebox.showerror(
                "Erro de Entrada", f"Dados da superfície inválidos.\n\nErro: {e}"
            )

    def add_superficie_bspline(self, nome, pontos_str, popup, cor):
        try:
            linhas_str = [
                linha.strip() for linha in pontos_str.split(";") if linha.strip()
            ]
            if len(linhas_str) < 4:
                raise ValueError("São necessárias pelo menos 4 linhas de pontos.")

            pontos_matriz = []
            num_cols = -1

            for i, linha_str in enumerate(linhas_str):
                # Adiciona colchetes para o ast.literal_eval entender como lista
                linha_pontos_raw = ast.literal_eval(f"[{linha_str}]")

                if i == 0:
                    num_cols = len(linha_pontos_raw)
                    if num_cols < 4:
                        raise ValueError(
                            "São necessárias pelo menos 4 colunas (pontos) por linha."
                        )
                elif len(linha_pontos_raw) != num_cols:
                    raise ValueError(
                        "Todas as linhas devem ter o mesmo número de pontos."
                    )

                linha_pontos_obj = [Ponto3D(x, y, z) for x, y, z in linha_pontos_raw]
                pontos_matriz.append(linha_pontos_obj)

            obj = SuperficieBSpline(
                nome if nome else f"superficie_bspline_{len(self.displayfile.objetos)}",
                pontos_matriz,
                cor,
            )
            self.displayfile.adicionar(obj)
            self.redesenhar()
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
            popup.destroy()

        except Exception as e:
            messagebox.showerror(
                "Erro de Entrada",
                f"Dados da superfície B-Spline inválidos.\nVerifique o formato (mín 4x4) e a sintaxe.\n\nErro: {e}",
            )

    def add_obj_3d(self, nome, vertices_str, arestas_str, popup, cor):
        try:
            vertices_raw = [
                ast.literal_eval(v.strip())
                for v in vertices_str.split(";")
                if v.strip()
            ]
            vertices = [Ponto3D(x, y, z) for x, y, z in vertices_raw]
            arestas = [
                ast.literal_eval(a.strip()) for a in arestas_str.split(";") if a.strip()
            ]
            if not vertices or not arestas:
                raise ValueError("Vértices ou arestas vazios.")
            obj = Objeto3D(
                nome if nome else f"obj3d_{len(self.displayfile.objetos)}",
                vertices,
                arestas,
                cor,
            )
            self.displayfile.adicionar(obj)
            self.redesenhar()
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
            popup.destroy()
        except Exception as e:
            messagebox.showerror(
                "Erro de Entrada", f"Dados do objeto 3D inválidos.\n\nErro: {e}"
            )

    def add_obj(self, nome, tipo, coords, popup, cor, preenchido=False):
        try:
            if tipo == TipoObjeto.CURVA.value:
                if len(coords) < 4 or (len(coords) - 4) % 3 != 0:
                    messagebox.showerror(
                        "Erro de Entrada",
                        "Uma curva de Bézier precisa de 4, 7, 10, ... pontos de controle.",
                    )
                    return
            if tipo == TipoObjeto.B_SPLINE.value:
                if len(coords) < 4:
                    messagebox.showerror(
                        "Erro de Entrada",
                        "Uma B-Spline precisa de no mínimo 4 pontos de controle.",
                    )
                    return
            coords_float = [(float(x), float(y)) for x, y in coords]
            obj = Objeto(
                nome if nome else f"{tipo}_{len(self.displayfile.objetos)}",
                tipo,
                coords_float,
                cor,
                preenchido,
            )
            self.displayfile.adicionar(obj)
            self.redesenhar()
            self.lista_objetos.insert(tk.END, f"{obj.nome} ({obj.tipo})")
            popup.destroy()
        except (ValueError, TypeError, SyntaxError) as e:
            messagebox.showerror(
                "Erro de Entrada",
                f"Coordenadas inválidas. Verifique o formato.\n\nErro: {e}",
            )

    def abrir_transformacoes_popup(self):
        if not self.selected_obj_name:
            messagebox.showerror("Erro", "Selecione um objeto para transformar.")
            return
        obj = self.displayfile.get_by_name(self.selected_obj_name)
        if not obj:
            return

        if isinstance(obj, (Objeto3D, SuperficieBezier, SuperficieBSpline)):
            self.abrir_transformacoes_popup_3d(obj)
        else:
            self.abrir_transformacoes_popup_2d()

    def fechar_popup_transformacao(self, popup):
        self.lista_objetos.bind("<<ListboxSelect>>", self.on_obj_select)
        popup.destroy()

    def abrir_transformacoes_popup_3d(self, obj):
        popup = tk.Toplevel(self)
        popup.title(f"Transformar {obj.nome}")
        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)
        frame_t = ttk.Frame(notebook)
        notebook.add(frame_t, text="Translação")
        dx_t, dy_t, dz_t = tk.Entry(frame_t), tk.Entry(frame_t), tk.Entry(frame_t)
        tk.Label(frame_t, text="dx:").grid(row=0, column=0)
        dx_t.grid(row=0, column=1)
        tk.Label(frame_t, text="dy:").grid(row=1, column=0)
        dy_t.grid(row=1, column=1)
        tk.Label(frame_t, text="dz:").grid(row=2, column=0)
        dz_t.grid(row=2, column=1)
        tk.Button(
            frame_t,
            text="Aplicar",
            command=lambda: self.aplicar_transformacao_3d(
                obj,
                "translacao",
                dx=float(dx_t.get() or 0),
                dy=float(dy_t.get() or 0),
                dz=float(dz_t.get() or 0),
                popup=popup,
            ),
        ).grid(row=3, columnspan=2)
        frame_s = ttk.Frame(notebook)
        notebook.add(frame_s, text="Escalonamento")
        sx_s, sy_s, sz_s = tk.Entry(frame_s), tk.Entry(frame_s), tk.Entry(frame_s)
        tk.Label(frame_s, text="sx:").grid(row=0, column=0)
        sx_s.grid(row=0, column=1)
        tk.Label(frame_s, text="sy:").grid(row=1, column=0)
        sy_s.grid(row=1, column=1)
        tk.Label(frame_s, text="sz:").grid(row=2, column=0)
        sz_s.grid(row=2, column=1)
        tk.Button(
            frame_s,
            text="Aplicar",
            command=lambda: self.aplicar_transformacao_3d(
                obj,
                "escalonamento",
                sx=float(sx_s.get() or 1),
                sy=float(sy_s.get() or 1),
                sz=float(sz_s.get() or 1),
                popup=popup,
            ),
        ).grid(row=3, columnspan=2)
        frame_r = ttk.Frame(notebook)
        notebook.add(frame_r, text="Rotação")
        angulo_r = tk.Entry(frame_r)
        eixo_var = tk.StringVar(value="y")
        tk.Label(frame_r, text="Ângulo (graus):").grid(row=0, column=0)
        angulo_r.grid(row=0, column=1)
        tk.Label(frame_r, text="Eixo:").grid(row=1, column=0)
        tk.Radiobutton(frame_r, text="X", variable=eixo_var, value="x").grid(
            row=1, column=1, sticky="w"
        )
        tk.Radiobutton(frame_r, text="Y", variable=eixo_var, value="y").grid(
            row=2, column=1, sticky="w"
        )
        tk.Radiobutton(frame_r, text="Z", variable=eixo_var, value="z").grid(
            row=3, column=1, sticky="w"
        )
        tk.Button(
            frame_r,
            text="Aplicar",
            command=lambda: self.aplicar_transformacao_3d(
                obj,
                "rotacao",
                angulo=float(angulo_r.get() or 0),
                eixo=eixo_var.get(),
                popup=popup,
            ),
        ).grid(row=4, columnspan=2)

    def abrir_transformacoes_popup_2d(self):
        self.lista_objetos.unbind("<<ListboxSelect>>")
        popup = tk.Toplevel(self)
        popup.title(f"Transformar {self.selected_obj_name}")
        popup.geometry("350x300")
        popup.protocol(
            "WM_DELETE_WINDOW", lambda: self.fechar_popup_transformacao(popup)
        )
        notebook = ttk.Notebook(popup)
        notebook.pack(expand=True, fill="both", padx=10, pady=10)
        frame_t = ttk.Frame(notebook)
        notebook.add(frame_t, text="Translação")
        dx_t, dy_t = tk.Entry(frame_t), tk.Entry(frame_t)
        tk.Label(frame_t, text="dx:").grid(row=0, column=0, sticky="w")
        dx_t.grid(row=0, column=1)
        tk.Label(frame_t, text="dy:").grid(row=1, column=0, sticky="w")
        dy_t.grid(row=1, column=1)
        tk.Button(
            frame_t,
            text="Aplicar",
            command=lambda: self.aplicar_transformacao(
                "translacao",
                dx=float(dx_t.get() or 0),
                dy=float(dy_t.get() or 0),
                popup=popup,
            ),
        ).grid(row=2, columnspan=2, pady=10)
        frame_s = ttk.Frame(notebook)
        notebook.add(frame_s, text="Escalonamento")
        sx_s, sy_s = tk.Entry(frame_s), tk.Entry(frame_s)
        tk.Label(frame_s, text="sx:").grid(row=0, column=0, sticky="w")
        sx_s.grid(row=0, column=1)
        tk.Label(frame_s, text="sy:").grid(row=1, column=0, sticky="w")
        sy_s.grid(row=1, column=1)
        tk.Button(
            frame_s,
            text="Aplicar",
            command=lambda: self.aplicar_transformacao(
                "escalonamento_natural",
                sx=float(sx_s.get() or 1),
                sy=float(sy_s.get() or 1),
                popup=popup,
            ),
        ).grid(row=2, columnspan=2, pady=10)
        frame_r = ttk.Frame(notebook)
        notebook.add(frame_r, text="Rotação")
        tk.Label(frame_r, text="Ângulo (graus):").grid(row=0, column=0, sticky="w")
        angulo_r = tk.Entry(frame_r)
        angulo_r.grid(row=0, column=1)
        rot_var = tk.StringVar(value="centro_objeto")
        tk.Radiobutton(
            frame_r, text="Centro do objeto", variable=rot_var, value="centro_objeto"
        ).grid(row=1, sticky="w")
        tk.Radiobutton(
            frame_r,
            text="Origem do mundo (0,0)",
            variable=rot_var,
            value="centro_mundo",
        ).grid(row=2, sticky="w")
        frame_arbitrario = tk.Frame(frame_r)
        frame_arbitrario.grid(row=3, sticky="w")
        tk.Radiobutton(
            frame_arbitrario, text="Ponto:", variable=rot_var, value="ponto_arbitrario"
        ).pack(side="left")
        x_r, y_r = tk.Entry(frame_arbitrario, width=5), tk.Entry(
            frame_arbitrario, width=5
        )
        tk.Label(frame_arbitrario, text="x:").pack(side="left")
        x_r.pack(side="left")
        tk.Label(frame_arbitrario, text="y:").pack(side="left")
        y_r.pack(side="left")
        tk.Button(
            frame_r,
            text="Aplicar",
            command=lambda: self.aplicar_transformacao(
                rot_var.get(),
                angulo=float(angulo_r.get() or 0),
                px=float(x_r.get() or 0),
                py=float(y_r.get() or 0),
                popup=popup,
            ),
        ).grid(row=4, columnspan=2, pady=10)

    def aplicar_transformacao_3d(self, obj, tipo_transf, **kwargs):
        popup = kwargs.pop("popup", None)
        try:
            if tipo_transf == "translacao":
                matriz = Transformacoes3D.get_matriz_translacao(
                    kwargs["dx"], kwargs["dy"], kwargs["dz"]
                )
                obj.aplicar_transformacao_3d(matriz)
            elif tipo_transf == "escalonamento":
                if isinstance(obj, Objeto3D):
                    cx, cy, cz = obj.get_centro_objeto_3d()
                elif isinstance(obj, (SuperficieBezier, SuperficieBSpline)):
                    if isinstance(obj, SuperficieBezier):
                        todos_pontos = obj.pontos_controle.flatten()
                        n_pontos = 16
                    else:  # SuperficieBSpline
                        todos_pontos = obj.pontos_controle_matriz.flatten()
                        n_pontos = obj.n_linhas * obj.n_cols

                    cx = sum(p.x for p in todos_pontos) / n_pontos
                    cy = sum(p.y for p in todos_pontos) / n_pontos
                    cz = sum(p.z for p in todos_pontos) / n_pontos

                t1 = Transformacoes3D.get_matriz_translacao(-cx, -cy, -cz)
                s = Transformacoes3D.get_matriz_escalonamento(
                    kwargs["sx"], kwargs["sy"], kwargs["sz"]
                )
                t2 = Transformacoes3D.get_matriz_translacao(cx, cy, cz)
                matriz = t2 @ s @ t1
                obj.aplicar_transformacao_3d(matriz)

            elif tipo_transf == "rotacao":
                angulo_rad = np.deg2rad(kwargs["angulo"])
                eixo = kwargs["eixo"]

                if isinstance(obj, Objeto3D):
                    cx, cy, cz = obj.get_centro_objeto_3d()
                elif isinstance(obj, (SuperficieBezier, SuperficieBSpline)):
                    if isinstance(obj, SuperficieBezier):
                        todos_pontos = obj.pontos_controle.flatten()
                        n_pontos = 16
                    else:  # SuperficieBSpline
                        todos_pontos = obj.pontos_controle_matriz.flatten()
                        n_pontos = obj.n_linhas * obj.n_cols

                    cx = sum(p.x for p in todos_pontos) / n_pontos
                    cy = sum(p.y for p in todos_pontos) / n_pontos
                    cz = sum(p.z for p in todos_pontos) / n_pontos

                t1 = Transformacoes3D.get_matriz_translacao(-cx, -cy, -cz)
                if eixo == "x":
                    r = Transformacoes3D.get_matriz_rotacao_x(angulo_rad)
                elif eixo == "y":
                    r = Transformacoes3D.get_matriz_rotacao_y(angulo_rad)
                else:
                    r = Transformacoes3D.get_matriz_rotacao_z(angulo_rad)
                t2 = Transformacoes3D.get_matriz_translacao(cx, cy, cz)
                matriz = t2 @ r @ t1
                obj.aplicar_transformacao_3d(matriz)

            self.redesenhar()
            if popup:
                popup.destroy()
        except Exception as e:
            messagebox.showerror("Erro", f"Entrada inválida: {e}")

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
                matriz = self.transformacoes.get_matriz_rotacao(
                    np.deg2rad(kwargs["angulo"])
                )
                self.transformacoes.aplicar_transformacao_generica(obj, matriz)
            elif tipo_transf == "centro_objeto":
                self.transformacoes.aplicar_rotacao_centro_objeto(
                    obj, np.deg2rad(kwargs["angulo"])
                )
            elif tipo_transf == "ponto_arbitrario":
                self.transformacoes.aplicar_rotacao_ponto_arbitrario(
                    obj, np.deg2rad(kwargs["angulo"]), kwargs["px"], kwargs["py"]
                )

            self.redesenhar()
            if "popup" in kwargs:
                self.fechar_popup_transformacao(kwargs["popup"])
        except (ValueError, TypeError) as e:
            messagebox.showerror(
                "Erro de Entrada",
                f"Entrada inválida. Verifique os valores.\n\nErro: {e}",
            )

    def pan(self, dx, dy):
        theta = np.deg2rad(self.viewport.window_angle)
        c, s = np.cos(theta), np.sin(theta)
        world_dx, world_dy = dx * c - dy * s, dx * s + dy * c
        self.viewport.wmin = (
            self.viewport.wmin[0] + world_dx,
            self.viewport.wmin[1] + world_dy,
        )
        self.viewport.wmax = (
            self.viewport.wmax[0] + world_dx,
            self.viewport.wmax[1] + world_dy,
        )
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
    root.title("SGI 2D/3D - Superfícies Bicúbicas (Bézier e B-Spline)")
    root.geometry("1200x800")
    app = App(root)
    app.mainloop()