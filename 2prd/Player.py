from typing import final
import numpy as np

from MovableObject  import MovableObject
from Rocket import Rocket
from NoPublicConstructor import NoPublicConstructor

@final # pip install mypy && mypy . -s for static type checking
class Player(MovableObject, metaclass=NoPublicConstructor):
    _instance=None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = type(cls).get_instance(cls, *args, **kwargs)
        return cls._instance

    def __init_subclass__(cls):
        raise TypeError("subclassing not supported")

    def __init__(self, _plt, name=None):
        super().__init__(_plt, name)
        self.name = f"PLR#{self.name}"

        self.geometry = self.generate_geometry()
        self.rockets = []
        self.rockets.append(Rocket(_plt))

    def activate_thrusters(self):
        print("activating thrusters")
        self.fire_rocket()
        self.update_movement()

    def fire_rocket(self):
        print("firing rocket (pop and use)")
        if not self.rockets:
            print("out of rockets")
            return

        rocket = self.rockets.pop() #fire the last rocket in list (remove from list and use it)
        rocket.update_movement(0.123)

    def update_movement(self, delta_time):
        print(f"child(player-{self.name}) update_movement")
        super().update_movement(delta_time)

    def generate_geometry(self):
        print(f"child(player-{self.name}) generating geometry")
        return np.identity(4)


if __name__ == "__main__":
    print("Testing player")
    player = Player.get_instance(None)
    print(f"user: {player.name} created")
    player.fire_rocket()

    player1 = Player.get_instance(None,"singleton")
    print(f"user: {player1.name} created")
    player.fire_rocket()

    if player != player1:
        raise Exception("singleton pattern broken")

    print("test passed")
    #print(f"user: {player.name} created")
    #player = Player(None, "user")
    #print(f"user: {player.name} created")