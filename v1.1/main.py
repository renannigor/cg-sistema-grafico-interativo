import tkinter as tk

from ui.app import App

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sistema Básico de CG 2D")
    root.geometry("800x600")
    app = App(root)
    app.mainloop()
