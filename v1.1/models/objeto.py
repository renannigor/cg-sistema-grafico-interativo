from typing import List, Tuple


class Objeto:
    def __init__(self, nome: str, tipo: str, coords: List[Tuple[float, float]]):
        self.nome = nome
        self.tipo = tipo
        self.coords = coords
