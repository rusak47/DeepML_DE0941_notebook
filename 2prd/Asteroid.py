from MovableObject import MovableObject
import numpy as np
import random as rnd

class Asteroid(MovableObject):

    def __init__(self, _plt, name = None):
        super().__init__(_plt, name)
        self.name = f"ASTER#{self.name}"
        self.color = 'b'

        self.geometry = self.generate_geometry()

    def update_movement(self, delta_time):
        print(f"child(asteroid-{self.name}) update_movement")
#        super().update_movement(delta_time)

    def generate_geometry(self):
        print(f"child(asteroid-{self.name}) update_movement")
        # generate random size and position
        size = rnd.random()
        offset = (-1 if size > 0.5 else 1) * rnd.random() * 9.8  # TODO respect canvas size

        return np.identity(2)*size + offset

