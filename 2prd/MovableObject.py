from abc import abstractmethod
from StaticObject import StaticObject
import numpy as np
import uuid

class MovableObject(StaticObject):

    def __init__(self, _plt, name = None):
        super().__init__(_plt)
        if not name:
            name = str(uuid.uuid4()).replace('-', '')[:8]
        self.name = name #for debugging purposes

        #self.plt = _plt
        self.geometry = []
        self.speed = 0
        self.angle = 0
        self.color = 'r'
        """
        'b' as blue
        'g' as green
        'r' as red
        'c' as cyan
        'm' as magenta
        'y' as yellow
        'k' as black
        'w' as white
        
        Copied from: Specifying colors — Matplotlib 3.10.6 documentation - <https://matplotlib.org/stable/users/explain/colors/colors.html>
        """

        self.pos = np.array([0,0])
        self.dir = np.array([0,1])
        self.C = np.identity(3)  # sq matrix with 1s diagonal
        self.R = np.identity(3)
        self.T = None  # what is this about?

    # TODO finish
    def draw(self):
        print(f"drawing {self.name}...")
        x_data = []
        y_data = []
        for vec2 in self.geometry:
            x_data.append(vec2[0])
            y_data.append(vec2[1])
        self.plt.plot(x_data, y_data, color=self.color, label=f"{self.name}")
        self.plt.text(x_data[0], y_data[0], f"{self.name}", size=8, rotation=10.,
         ha="center", va="center"
         )

    def update_movement(self, delta_time):
        print("child(MovableObj) update_movement")
        #super().update_movement(delta_time)

    @abstractmethod
    def generate_geometry(self):
        raise Exception("Child specific method - implementing is required")