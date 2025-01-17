import math
import sys

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

class BoundingBox:
    def __init__(self, x_, y_, w_, h_):
        self.x = x_
        self.y = y_
        self.w = w_
        self.h = h_

    def contains(self, x, y, r) -> bool:
        if (x >= self.x) and (x <= self.x + self.w) and (y <= self.y+self.h) and (y>= self.y):
            return True
        else:
            du = abs(y - self.y)
            dd = abs(y - self.y - self.h)
            dl = abs(x - self.x)
            dr = abs(x - self.x - self.w)

            md = min(du, dd, dl, dr)
            if md <= r:
                return True
            else:
                return False


class quad_tree:
    def __init__(self, boundary_: BoundingBox, cap: int):
        self.boundary = boundary_
        self.capacity = cap
        self.elements: list[Object] = []
        self.divided=False

    def subdivide(self):
        #reference
        '''
                |
            q2  |   q1
                |
        ------------------
                |
            q3  |     q4
                |
        '''

        q1b = BoundingBox(self.boundary.x + self.boundary.w/2, self.boundary.y, self.boundary.w/2, self.boundary.h/2)
        q2b = BoundingBox(self.boundary.x, self.boundary.y, self.boundary.w/2, self.boundary.h/2)
        q3b = BoundingBox(self.boundary.x, self.boundary.y + self.boundary.h/2, self.boundary.w/2, self.boundary.h/2)
        q4b = BoundingBox(self.boundary.x + self.boundary.w/2, self.boundary.y + self.boundary.h/2, self.boundary.w/2, self.boundary.h/2)

        self.q1 = quad_tree(q1b, self.capacity)
        self.q2 = quad_tree(q2b, self.capacity)
        self.q3 = quad_tree(q3b, self.capacity)
        self.q4 = quad_tree(q4b, self.capacity)

        self.divided=True

        for p in self.elements:
            self.q1.insert(p)
            self.q2.insert(p)
            self.q3.insert(p)
            self.q4.insert(p)
        self.elements.clear()


    def insert(self, obj:Object):
        if self.boundary.contains(obj.pos.x, obj.pos.y, obj.radius):
            if (len(self.elements) < self.capacity) and not self.divided:
                self.elements.append(obj)
            else:
                if not self.divided:
                    self.subdivide()
                    self.q1.insert(obj)
                    self.q2.insert(obj)
                    self.q3.insert(obj)
                    self.q4.insert(obj)
                else:
                    self.q1.insert(obj)
                    self.q2.insert(obj)
                    self.q3.insert(obj)
                    self.q4.insert(obj)

    def get_neighbours(self, obj:Object) -> list[Object]:
        #return all the elements of the deepest quadtree possible where this point is found
        if (obj) in self.elements:
            return self.elements
        elif self.divided:
            l1 = self.q1.get_neighbours(obj)
            l2 =self.q2.get_neighbours(obj)
            l3 =self.q3.get_neighbours(obj)
            l4 =self.q4.get_neighbours(obj)

            rl = l1 + l2 + l3 + l4
            return rl

        else:
            return []


class VerletSolver:
    def __init__(self, step: int, objs: list[Object], acenter: Vec2, arad: float):
        self.gravity: Vec2 = Vec2(0.0, 2000.0)
        self.objects = objs
        self.area_center = acenter
        self.area_radius = arad
        self.steps = step

    def update(self, qt: quad_tree):
        self.apply_gravity()
        self.apply_constraints()
        self.solve_collisions(qt)
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


    def solve_collisions(self, qt: quad_tree):
        for obj in self.objects:
            for _obj in qt.get_neighbours(obj):
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
