import pygame
from enum import Enum
import random
import colorsys

pygame.init()

width, height = 720, 480

window = pygame.display.set_mode((width, height))

pygame.display.set_caption("falling sand")

rect_surface = pygame.Surface((width, height))
rect_surface.fill((30, 30, 30))

square_size = 4
grid_color = (30, 30, 30)
grid_width, grid_height = int(width/square_size), int(height/square_size)

class State(Enum):
    VOID = 0
    SAND = 1
    WATER = 2
    ROCK = 3


gravity = 0.7  # blocks per second
max_vel = 6.0
class Particle:
    def __init__(self, state:State):
        self.state = state
        self.velocity = 1.0

    def reset_vel(self):
        self.velocity = 1.0

class Sand(Particle):
    def __init__(self, col):
        super().__init__(State.SAND)
        self.color = col

class Void(Particle):
    def __init__(self):
        super().__init__(State.VOID)
        self.color = (30, 30, 30)

class Rock(Particle):
    def __init__(self, col):
        super().__init__(State.ROCK)
        self.color = col

grid = [[Void() for _ in range(grid_width)] for _ in range(height)]

change_list = []

def fast_random_sign():
    return 1 if random.getrandbits(1) == 1 else -1

def simulate(state_grid):
    change_list.clear()
    for i in range(grid_width, -1, -1):
        for j in range(grid_height, -1, -1):
            k = fast_random_sign()

            if(grid[i][j].state == State.SAND and j + 1 < grid_height):
                if (grid[i][j+1].state == State.VOID ):
                    if(grid[i][j].velocity <= max_vel):
                        grid[i][j].velocity += gravity
                    for x in range(round(grid[i][j].velocity), 0, -1):
                        if j+x < grid_height:
                            if(grid[i][j+x].state == State.VOID):
                                grid[i][j], grid[i][j + x] = grid[i][j + x], grid[i][j]
                                change_list.append([i, j])
                                change_list.append([i, j + x])

                elif(grid[i+k][j+1].state == State.VOID and i + k < grid_width and i-k > -1):
                    grid[i][j].reset_vel()
                    grid[i][j], grid[i+k][j + 1] = grid[i+k][j + 1], grid[i][j]
                    change_list.append([i,j])
                    change_list.append([i+k,j + 1])


                elif (grid[i - k][j + 1].state == State.VOID and i-k > -1 and i + k < grid_width):
                    grid[i][j].reset_vel()
                    grid[i][j] , grid[i - k][j + 1]= grid[i - k][j + 1], grid[i][j]
                    change_list.append([i,j])
                    change_list.append([i-k,j + 1])

                else:
                    grid[i][j].reset_vel()


            
brush_size = 2

def randomize_lightness(rgb):
    # Convert RGB to HSL
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Randomly adjust the lightness
    lightness_adjustment = random.uniform(-0.1, 0.1)  # adjust this range to your liking
    l += lightness_adjustment
    l = max(0, min(l, 1))  # clamp lightness to [0, 1] range

    # Convert HSL back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # Return the new RGB values as integers
    return int(r * 255), int(g * 255), int(b * 255)


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    window.fill((10, 10, 10))

    simulate(grid)

    mouse_pos = pygame.mouse.get_pos()

    for col in range(0, width, square_size):
        i = int(col/square_size)
        for row in range(0, height, square_size):
            j = int(row/square_size)

            rect = pygame.Rect(col, row, square_size, square_size)

            if pygame.mouse.get_pressed()[0]:
                if(rect.collidepoint(mouse_pos)):
                    for x in range(-brush_size, brush_size):
                        for y in range(-brush_size, brush_size):
                            if(grid[i+x][j+y].state == State.VOID):
                                if(random.uniform(0.0, 1.0) > 0.4):
                                    grid[i + x][j+y] = Sand(randomize_lightness((214, 193, 116)))
                                    change_list.append([i+x,j+y])

            if pygame.mouse.get_pressed()[2]:
                if(rect.collidepoint(mouse_pos)):
                    for x in range(-brush_size, brush_size):
                        for y in range(-brush_size, brush_size):
                            grid[i + x][j+y] = Rock(randomize_lightness((150, 150, 150)))
                            change_list.append([i+x,j+y])

    for el in change_list:
        x, y= el[0], el[1]
        grid_color = grid[x][y].color
        rect = pygame.Rect(x*square_size, y*square_size, square_size, square_size)
        pygame.draw.rect(rect_surface, grid_color, rect)

    window.blit(rect_surface, (0, 0))

    # Update the display
    pygame.display.flip()
# Quit pygame
pygame.quit()

