from StaticObject import StaticObject

class Planet(StaticObject):
    def __init__(self, _plt):
        super().__init__(_plt)
        #self.plt = _plt

    def update_movement(self, delta_time):
        print("child(planet) update_movement")
        super().update_movement(delta_time)
