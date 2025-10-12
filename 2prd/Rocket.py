from MovableObject import MovableObject
import numpy as np

class Rocket(MovableObject):
    def __init__(self, _plt):
        super().__init__(_plt)
        #self.plt = _plt
        self.geometry = self.generate_geometry()

    def update_movement(self, delta_time):
        print("child(rocket) update_movement")
        super().update_movement(delta_time)

    def generate_geometry(self):
        print("child(rocket) generate_geometry")
        return np.identity(1)
