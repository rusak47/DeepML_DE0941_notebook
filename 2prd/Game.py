import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()

from typing import final
import numpy as np
from Player import Player
from Asteroid import Asteroid

from MovableObject  import MovableObject
from NoPublicConstructor import NoPublicConstructor

@final
class Game(metaclass=NoPublicConstructor):
    _instance=None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = type(cls).get_instance(cls, *args, **kwargs)
        return cls._instance

    def __init_subclass__(cls):
        raise TypeError("subclassing not supported")

    def __init__(self, _plt, playername=None):
        self._plt = _plt

        self.player = Player.get_instance(self._plt,playername)

        self.actors = [self.player, Asteroid(self._plt), Asteroid(self._plt)]
        self.lives = 1
        self.score = 0

        self.is_running = False

        fig, _ = _plt.subplots()
        fig.canvas.mpl_connect('key_press_event', self._on_press)

        self.main()

    def _on_press(self, event):
        print(f"event {event.key} {self.player.angle}")
        if event.key == 'escape':
            self.is_running = False
        elif event.key == 'left':
            self.player.angle += 5
        elif event.key == 'right':
            self.player.angle -= 5

    def main(self):
        self.run()

        # cleanup
        self._plt.close('all')

    def run(self):
        self.is_running = True
        while self.is_running:
            self._plt.clf()
            self._plt.xlim(-10, 10)
            self._plt.ylim(-10, 10)

            for each in self.actors:
                each.draw()
                # TODO display angle in plot title
                self._plt.title(f"angle: {self.player.angle}")

            self._plt.draw()
            self._plt.pause(1e-3)

if __name__ == "__main__":
    print("Testing game")
    #game = Game()
    game0 = Game.get_instance(plt, "MCC")
    game1 = Game.get_instance()

    print(f"game main character name: {Game.get_instance().player.name}")
    print(f"game characters count: {len(Game.get_instance().actors)}")
    print(f"game characters count: {Game.get_instance().is_running}")

    if game0 != game1:
        raise Exception("singleton pattern broken")

    print("test passed")