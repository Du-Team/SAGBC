class Edge:
    def __init__(self, u, v, weight=0.0):
        self.u = u
        self.v = v
        self.weight = weight
        self.is_core = False

    def set_core(self, bool_value):

        self.is_core = bool_value

    def __repr__(self):
        return f"Edge({self.u}, {self.v}, weight={self.weight:.4f})"

    def __eq__(self, other):
        if isinstance(other, Edge):
            return (self.u == other.u and self.v == other.v) or (self.u == other.v and self.v == other.u)
        return False

    def __hash__(self):
        return hash(tuple(sorted([self.u, self.v])))
