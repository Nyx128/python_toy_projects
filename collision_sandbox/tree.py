class BoundingBox:
    def __init__(self, x_, y_, w_, h_):
        self.x = x_
        self.y = y_
        self.w = w_
        self.h = h_

    def contains(self, x, y) -> bool:
        if (x >= self.x) and (x <= self.x + self.w) and (y <= self.y+self.h) and (y>= self.y):
            return True
        else:
            return False


class quad_tree:
    def __init__(self, boundary_: BoundingBox, cap: int):
        self.boundary = boundary_
        self.capacity = cap
        self.elements = []
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
            self.q1.insert(p[0], p[1])
            self.q2.insert(p[0], p[1])
            self.q3.insert(p[0], p[1])
            self.q4.insert(p[0], p[1])
        self.elements.clear()


    def insert(self, x, y):
        if self.boundary.contains(x, y):
            if (len(self.elements) < self.capacity) and not self.divided:
                self.elements.append((x, y))
            else:
                if not self.divided:
                    self.subdivide()
                    self.q1.insert(x, y)
                    self.q2.insert(x, y)
                    self.q3.insert(x, y)
                    self.q4.insert(x, y)
                else:
                    self.q1.insert(x, y)
                    self.q2.insert(x, y)
                    self.q3.insert(x, y)
                    self.q4.insert(x, y)

    def get_neighbours(self, x, y) -> tuple[bool, list]:
        #return all the elements of the deepest quadtree possible where this point is found
        if (x, y) in self.elements:
            return True, self.elements
        elif self.divided:
            _, l1 = self.q1.get_neighbours(x, y)
            _, l2 =self.q2.get_neighbours(x, y)
            _, l3 =self.q3.get_neighbours(x, y)
            _, l4 =self.q4.get_neighbours(x, y)

            rl = l1 + l2 + l3 + l4
            if len(rl) == 0:
                return False, []
            else:
                return True, rl

        else:
            return False, []
