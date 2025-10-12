from __future__ import annotations

import math
from warnings import catch_warnings

#from telnetlib import GA
import numpy as np
import os
import matplotlib
import matplotlib.backend_bases

if os.name == "darwin":
    matplotlib.use("MacOSX")  # for mac
else:
    matplotlib.use("TkAgg")  # for unix/windows

import matplotlib.pyplot as plt

SPACE_SIZE = (9, 9)
plt.rcParams["figure.figsize"] = (15, 15)
plt.ion()  # interactive mode
plt.style.use("dark_background")



def rotation_mat(degrees):
    radians = np.deg2rad(degrees)
    R = np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians), np.cos(radians), 0],
        [0,0,1]
    ])
    return R


def translation_mat(dx, dy):
    #task 2.5
    T = np.array([
        [1,0,dx],
        [0,1,dy],
        [0,0,1]
    ])
    return T


def scale_mat(sx, sy):
    # task 2.6
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    return S


def skew_mat(x, y):
    #task2.6a/actually task2.8
    kx = 0.2 #np.tan(x)
    ky = 0.4 #np.tan(y)
    W = np.array([
        [1, kx, 0],
        [ky, 1, 0],
        [0, 0, 1]
    ])

    return dot(W, np.array([x,y,1]))


def dot(X, Y):
    product = np.dot(X, Y)
    return product


def vec2d_to_vec3d(vec2):
    # 2.4 matrix operations
    # transform shape 1,2 to 1,3
    I = np.identity(3) #first create the biggest dimension matrix
    #and then take the necessary dimensions

    vec3 = dot(vec2, I[:2,:]) # :2 <- from 0 to 2, : <- from the start till the end
    #np.array([vec2[0], vec2[1], 0])) <- the result we want achieve

    # we need to enable z axis for movement (as per 2.4 task)
    vec3 += np.array([0, 0, 1])
    return vec3


def vec3d_to_vec2d(vec3):
    # 2.4 matrix operations
    # transform shape 1,3 to 1,2
    I = np.identity(3)  # first create the biggest dimension matrix
    # and then take the necessary dimensions

    vec2 = dot(vec3, I[:, :2])
    #np.array([vec3[0], vec3[1]]))
    return vec2

def generateMidPointCirce(R, xc, yc):
    geometry = []
    x = 0
    y = R
    p = 1 - R
    while x <= y:
        #Q1
        geometry.append([xc+x,yc+y])
        geometry.append([xc+y,yc+x])
        #Q2
        geometry.append([xc-x,yc+y])
        geometry.append([xc-y,yc+x])
        #Q3
        geometry.append([xc-x,yc-y])
        geometry.append([xc-y,yc-x])
        #Q4
        geometry.append([xc+x,yc-y])
        geometry.append([xc+y,yc-x])

        # after some time of experimenting i came up with these changes to get as a result round asteroids
        x+=.1
        p += 2*x + .1
        if p >= 0:
            y-=.1
            p -= 2*y
    #geometry.sort(key=lambda pt: pt[0])
    return geometry

class StaticObject:
    def __init__(self, vec_pos):
        self.vec_pos: np.ndarray = vec_pos.astype(float)
        self.vec_dir_init = np.array([0.0, 1.0])
        self.vec_dir = np.array(self.vec_dir_init)
        self.geometry: list[np.ndarray] = []
        self.__angle: float = 0
        self.color = 'r'

        self.T_center = np.identity(3)
        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)
        self.T = np.identity(3)

        self.update_transformation()

    def set_angle(self, angle):
        self.__angle = angle
        self.R = rotation_mat(angle)

        vec3d = vec2d_to_vec3d(self.vec_dir_init)
        vec3d = dot(self.R, vec3d)
        self.vec_dir = vec3d_to_vec2d(vec3d)

        self.update_transformation()

    def get_angle(self) -> float:
        return self.__angle

    def update_movement(self, delta_time: float):
        pass

    def update_transformation(self):
        self.T = translation_mat(self.vec_pos[0], self.vec_pos[1])

        # 2.5 chain together R, T, S in C using correct order!
        #task 2.5. 1) c=I; 2) c=rc 3) c = tc 4) v=cv
        self.C = np.identity(3) #1)
        self.C = np.dot(self.S, self.C) #?) unlisted, yet important for rotation around own axis
        self.C = np.dot(self.R, self.C) #2) enable rotation
        self.C = np.dot(self.T, self.C) #3) enable movement


    def draw(self):
        x_values = []
        y_values = []

        for vec2d in self.geometry:

            vec3d = vec2d_to_vec3d(vec2d)
            vec3d = dot(self.C, vec3d)
            vec2d = vec3d_to_vec2d(vec3d)

            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        plt.plot(x_values, y_values, c=self.color)


class Planet(StaticObject):

    def __init__(self, vec_pos: np.ndarray = None):
        super().__init__(vec_pos)

        self.gravity_c = 1e2 # gravity force

        self.color = 'w'
        self.vec_F = np.zeros(2,) # planet gravity force
        self.radius = 1.0
        #self.geometry = []
        self.generate_geometry()

        self.update_transformation()

    def generate_geometry(self):
        # task2.7 generate planet geometry
        #data_rads = np.linspace(0, 2 * np.pi, 20)
        # todo task2.9 Mid-Point Circle Drawing Algorithm -> generate_geometry
        #x = self.radius * np.cos(data_rads)
        #y = self.radius * np.sin(data_rads)
        #for i in range(len(x)):
        #    self.geometry.append([x[i], y[i]])
        self.geometry = generateMidPointCirce(self.radius, 0, 0)

    def update_movement(self, delta_time: float):
        # task 2.8 gravity calculation with regard to player
        # 1) distance = sqrt((pos_planet - pos_player)**) <- is that correct? square root clears root inside it
        #distance = np.sqrt((self.vec_pos - Player.get_instance().vec_pos)*(self.vec_pos + Player.get_instance().vec_pos))
        distance = self.vec_pos - Player.get_instance().vec_pos

        # simple gravity as per task
        #self.vec_F = self.gravity_c/ (distance*distance)

        # some fancy gravity formula -> more dynamic in action
        F = self.gravity_c / np.sum(np.abs(distance))
        self.vec_F = distance / np.sqrt(np.sum(distance ** 2))
        self.vec_F *= F

        # distance(a,b) < a_r+b_r
        if (dot(distance, np.transpose(distance)) < (self.radius + Player.get_instance().radius)):
            try:
                Player.get_instance().minus_live()
            except:
                Game.get_instance().minus_live()

            print("collision detected")


class MovableObject(StaticObject):
    def __init__(self, vec_pos):
        super().__init__(vec_pos)
        self.speed = 0

    def update_movement(self, delta_time: float):
        self.vec_pos += self.vec_dir * self.speed * delta_time
        self.update_transformation()

        if abs(self.vec_pos[0]) > SPACE_SIZE[0]:
            self.vec_pos[0] = -self.vec_pos[0]/abs(self.vec_pos[0]) * SPACE_SIZE[0]
        if abs(self.vec_pos[1]) > SPACE_SIZE[1]:
            self.vec_pos[1] = -self.vec_pos[1]/abs(self.vec_pos[1]) * SPACE_SIZE[1]


class Asteroid(MovableObject):
    def __init__(self, vec_pos):
        super().__init__(vec_pos)

        self.color = 'g'

        step = 2 * np.pi / 20
        self.radius = np.random.random() * 0.2 + 0.2
        #self.geometry = [] #  generate asteroid geometry
        self.generate_geometry()

        self.S = np.identity(3) # TODO scew asteroid

        self.speed = np.random.random() * 20 + 10
        self.set_angle(np.random.random() * 360)
        self.update_transformation()

    def destroy(self):
        #todo some fancy animation
        self.color = 'b'
        self.speed = 0
        self.radius = 0

    def update_movement(self, delta_time: float):
        super().update_movement(delta_time)

    def generate_geometry(self):
        self.geometry = []
        data_rads = np.linspace(0, 2 * np.pi, 20)
        x = self.radius * np.cos(data_rads)
        y = self.radius * np.sin(data_rads)
        for i in range(len(x)):
            k = np.random.randint(0, len(x))+1
            if i % k == 0:
                _tuple = vec3d_to_vec2d(skew_mat(x[i],y[i]))
            else:
                _tuple = [x[i],y[i]]
            self.geometry.append(_tuple)

class Rocket(MovableObject):
    def __init__(self, vec_pos):
        super().__init__(vec_pos)

        self.color = 'y'

        self.geometry = np.array([
            [0, -0.1],
            [0, 0.1],
        ])
        self.radius = 0.1
        self.pos = np.array(Player.get_instance().vec_pos)
        self.speed = 60 + Player.get_instance().speed
        self.set_angle(Player.get_instance().get_angle())
        self.update_transformation()

    def update_movement(self, delta_time: float):
        super().update_movement(delta_time)

        for actor in Game.get_instance().actors:
            if type(actor) is Asteroid:
                distance = self.vec_pos - actor.vec_pos

                if (dot(distance, np.transpose(distance)) < (self.radius + actor.radius)):
                    print("asteroid collision with rocket detected")
                    Game.get_instance().actors.remove(actor)
                    Game.get_instance().add_points()
                    actor.destroy()


class Player(MovableObject):
    _instance: Player = None

    def __init__(self, vec_pos):
        super().__init__(vec_pos)
        self.geometry = np.array([
            # 2.3.
            [0,1],
            [1,0],
            [-1,0],
            [0,1]
        ])

        self.lives = 1
        self.radius = 1

        #task 2.5.
        #offset triangle to rotate around own axis
        offset = translation_mat(0, -0.5)
        self.S = dot(offset, self.S)

        #task 2.6
        self.S = scale_mat(0.5, 0.7)

        #task2.6a
        # todo skew on some angles
        #self.S = scew_mat(0.4, 0.3)

        self.speed = 0
        self.update_transformation()

        if not Player._instance:
            Player._instance = self
        else:
            raise Exception("Cannot construct singleton twice")

    def get_lives(self):
        return self.lives

    def minus_live(self):
        if not self.is_alive():
            raise Exception("Player have no lives")

        self.lives -= 1
        if self.lives <= 0:
            raise Exception("Player lost last life")


    def is_alive(self):
        return self.lives > 0

    def activate_thrusters(self):
        if not self.is_alive():
            self.speed = 0
            return

        if self.speed < 200:
            self.speed += 50.0

    def activate_brake(self):
        if not self.is_alive():
            self.speed = 0
            return

        if self.speed < 25:
            return
        self.speed -= 25.0

    def fire_rocket(self):
        if not self.is_alive():
            return
        rocket = Rocket(self.vec_pos)
        Game.get_instance().actors.append(rocket)

    def update_movement(self, delta_time: float):
        if not self.is_alive():
            self.speed = 0
            return

        self.speed -= delta_time * 30.0
        self.speed = max(0, self.speed)

        for actor in Game.get_instance().actors:
            if isinstance(actor, Planet):
                self.vec_pos += actor.vec_F * delta_time

        super().update_movement(delta_time)

        for actor in Game.get_instance().actors:
            if type(actor) is Asteroid :
                distance = self.vec_pos - actor.vec_pos

                if (dot(distance, np.transpose(distance)) < (self.radius + actor.radius)):
                    print("player collision with asteroid detected")
                    try:
                        Player.get_instance().minus_live()
                    except:
                        Game.get_instance().minus_live()

    @staticmethod
    def get_instance() -> Player:
        if not Player._instance:
            Player()
        return Player._instance


class Game:
    _instance: Game = None

    def game_over(self):
        print("game over")
        plt.clf()
        plt.axis("off")

        plt.title(
            f"GAME OVER; angle: {Player.get_instance().get_angle()} score: {self.score} speed: {round(Player.get_instance().speed, 1)} pos:  {Player.get_instance().vec_pos}")
        plt.draw()
        plt.pause(1e-3)
        self.is_running = False
        #todo handle events separately from gui updates


    def add_points(self, pt=10):
        self.score += pt

    def minus_live(self):
        if self.lives <= 0:
            self.game_over()
            return

        self.lives -= 1
        if self.lives <= 0:
            self.game_over()
            return

    def __init__(self):
        super(Game, self).__init__()
        self.is_running = True
        self.score = 0
        self.lives = 1

        self.actors: list[StaticObject] = [
            Player(vec_pos=np.array([0, 0])),
            Planet(vec_pos=np.array([-7, -3])),
            Planet(vec_pos=np.array([8, -4])),
        ]

        for _ in range(5):
            asteroid = Asteroid(vec_pos=np.array(
                [
                    np.random.randint(-SPACE_SIZE[0], SPACE_SIZE[0]),
                    np.random.randint(-SPACE_SIZE[1], SPACE_SIZE[1]),
                ]
            ))
            self.actors.append(asteroid)

        if not Game._instance:
            Game._instance = self
        else:
            raise Exception("Cannot construct singleton twice")

    def press(self: Game, event: matplotlib.backend_bases.Event):
        player = Player.get_instance()
        print("press", event.key)
        if event.key == "escape":
            self.is_running = False
            plt.close('all')

        elif event.key == "right":
            player.set_angle(player.get_angle() - 5)
        elif event.key == "left":
            player.set_angle(player.get_angle() + 5)
        elif event.key == "up":
            player.activate_thrusters()
        elif event.key == "down":
            player.activate_brake()
        elif event.key == " ":
            player.fire_rocket()

    def on_close(self: Game, event: matplotlib.backend_bases.Event):
        self.is_running = False

    def main(self: Game):

        fig, _ = plt.subplots()
        fig.canvas.mpl_connect("key_press_event", self.press)
        fig.canvas.mpl_connect("close_event", self.on_close)
        dt = 1e-3

        while self.is_running:
            plt.clf()
            plt.axis("off")

            plt.title(f"lives: {Player.get_instance().get_lives()}; angle: {Player.get_instance().get_angle()} score: {self.score} speed: {round(Player.get_instance().speed, 1)} pos:  {Player.get_instance().vec_pos}")
            plt.tight_layout(pad=0)

            plt.xlim(-10, 10)
            plt.ylim(-10, 10)

            for actor in self.actors:  # polymorhism
                actor.update_movement(dt)
                actor.draw()

            plt.draw()
            plt.pause(dt)

    @staticmethod
    def get_instance() -> Game:
        if not Game._instance:
            Game()
        return Game._instance


game = Game.get_instance()
game.main()