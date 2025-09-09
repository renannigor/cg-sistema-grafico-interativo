from models.objeto import Objeto


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
