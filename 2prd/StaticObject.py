from abc import ABC, abstractmethod
import numpy as np

class StaticObject(ABC):

    def __init__(self, _plt):
        self.plt = _plt # 'pointer' to main plt object
        self.pos = np.array([0,0]) #vec_pos
        self.dir_init = np.array([0,0]) #vec_dir vec_dir_init
        self.dir = None #vec_dir_init

        self.__angle = 0.0 #private property with name mangling (effectively says -  Doesn't override A.__attr)
                            #  prevent subclasses from overriding them. If your class is intended to be subclassed,
                            #  and it has attributes that you don’t want subclasses to use, then consider naming them with double leading underscores.
                            #
                            # Copied from: Single and Double Underscores in Python Names – Real Python - <https://realpython.com/python-double-underscore/#double-leading-underscore-in-classes-pythons-name-mangling>

    def set_angle(self, angle):
        if not angle:
            print("wrong angle")
        print(f"set angle {angle}")
        self.__angle = angle

    def get_angle(self):
        return self.__angle

    @abstractmethod
    def update_movement(self, delta_time):
        #todo any static movement?
        raise Exception("Child specific method - implementing is required")
