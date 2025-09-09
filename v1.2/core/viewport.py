from models.display_file import DisplayFile
from models.tipo_objeto import TipoObjeto


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

        for obj in displayfile.listar():
            coords_vp = [transform(x, y) for (x, y) in obj.coords]

            if obj.tipo == TipoObjeto.PONTO.value:
                x, y = coords_vp[0]
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=obj.cor)

            elif obj.tipo == TipoObjeto.RETA.value:
                # coords_vp é [(x1,y1),(x2,y2)] -> create_line espera valores planos
                flat = [coord for p in coords_vp for coord in p]
                if len(flat) >= 4:
                    self.canvas.create_line(*flat, fill=obj.cor, width=2)

            elif obj.tipo == TipoObjeto.WIREFRAME.value:
                if len(coords_vp) > 1:
                    for i in range(len(coords_vp) - 1):
                        p1 = coords_vp[i]
                        p2 = coords_vp[i + 1]
                        self.canvas.create_line(p1, p2, fill=obj.cor, width=2)

                    p_last = coords_vp[-1]
                    p_first = coords_vp[0]
                    self.canvas.create_line(p_last, p_first, fill=obj.cor, width=2)
