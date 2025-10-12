from abc import ABCMeta
from StaticObject import StaticObject
class NoPublicConstructor(ABCMeta):
    def __call__(cls, *args, **kwargs):
        raise TypeError("no public constructor")

    #def __create(self, *args, **kwargs):
    #    return super().__call__(*args, **kwargs)

    @classmethod
    def get_instance(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class Base(StaticObject, metaclass=NoPublicConstructor):
    _instance = None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if not cls._instance:
            #cls._instance = cls._NoPublicConstructor__create(*args, **kwargs)
            cls._instance = type(cls).get_instance(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, _plt, name=None):
        self.name = name

    def update_movement(self, delta_time):
        pass



"""
Term	                                            What it does
Metaclass (type, or your NoPublicConstructor)	    Creates classes (not instances)
Class (MyClass)                                    	Creates instances
Instance (obj = MyClass(...))	                    An actual object in memory

Copied from: ChatGPT - <https://chatgpt.com/>"""
if __name__ == "__main__":
    #obj = NoPublicConstructor("",(1,2),dict()) # doesnt work like that
    # This will raise:
    #Base()
    #Base(None,"")
    # TypeError: no public constructor

    #obj0 = Base._NoPublicConstructor__create(None)
    obj = Base.get_instance(None,"t1")
    print(obj.name)
    obj2 = Base.get_instance(None,"t2")
    #obj = Base._create(None)
    print(obj.name)
    print(obj == obj2)