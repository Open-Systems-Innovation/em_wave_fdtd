class Grid:
    """
    Computational grid for the FDTD simulation.
    """
    def __init__(self, nx, ny, dx, dy):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
