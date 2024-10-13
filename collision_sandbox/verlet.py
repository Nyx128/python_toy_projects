import math

class Vec2:
    def __init__(self, _x:float, _y:float):
        self.x=_x
        self.y =_y

    def __add__(self, other):
        return Vec2(self.x + other.x , self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x , self.y - other.y)

    def __mul__(self, other:float):
        return Vec2(self.x * other, self.y * other)

    def length(self):
        return math.sqrt((self.x * self.x) + (self.y * self.y))

    def sqlength(self):
        return self.x * self.x + self.y * self.y



class Object:
    def __init__(self, _pos:Vec2, rad:float):
        self.pos = _pos
        self.last_pos = _pos
        self.acc:Vec2 = Vec2(0.0, 0.0)
        self.radius = rad

    def update(self, dt: float):
        v = self.pos - self.last_pos
        self.last_pos= self.pos
        self.pos = self.pos + v + (self.acc * dt * dt)
        self.acc = Vec2(0.0, 0.0)

    def accelerate(self, _acc: Vec2):
        self.acc += _acc

    def setVelocity(self, vel:Vec2, dt:float):
        self.pos = self.last_pos + vel * dt


class VerletSolver:
    def __init__(self, step: int, objs: list[Object], acenter: Vec2, arad: float):
        self.gravity: Vec2 = Vec2(0.0, 2000.0)
        self.objects = objs
        self.area_center = acenter
        self.area_radius = arad
        self.steps = step

    def update(self):
        self.apply_gravity()
        self.apply_constraints()
        self.solve_collisions()
        self.update_positions(1.0/float(self.steps))

    def update_positions(self, dt:float):
        for o in self.objects:
            o.update(dt)

    def apply_gravity(self):
        for o in self.objects:
            o.accelerate(self.gravity)

    def apply_constraints(self):
        for obj in self.objects:
            to_obj = obj.pos - self.area_center
            dst = to_obj.length()
            if dst > self.area_radius - obj.radius:
                n = to_obj * (1.0/dst)
                obj.pos = self.area_center + n * (self.area_radius-obj.radius)


    def solve_collisions(self):
        for obj in self.objects:
            for _obj in self.objects:
                if obj != _obj:
                    col_axis = obj.pos - _obj.pos
                    dst = col_axis.sqlength()
                    check_dst = obj.radius + _obj.radius
                    if(dst < check_dst*check_dst):
                        dst = math.sqrt(dst)
                        n = col_axis * (1.0/dst)
                        delta = check_dst - dst
                        obj.pos+= n * delta * 0.5
                        _obj.pos -= n * delta * 0.5
