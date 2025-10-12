import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()

from Player import Player
from Asteroid import Asteroid
#todo singleton pattern for game

player = Player.get_instance(plt, "MC")
characters = [player, Asteroid(plt), Asteroid(plt)]

is_running = True

def on_press(event):
    global is_running, player
    print(f"event {event.key} {player.angle}")
    if event.key == 'escape':
        is_running = False
    elif event.key == 'left':
        player.angle += 5
    elif event.key == 'right':
        player.angle -= 5

fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)

while is_running:
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    for each in characters:
        each.draw()
        #TODO display angle in plot title
        plt.title(f"angle: {player.angle}")

    plt.draw()
    plt.pause(1e-3)
