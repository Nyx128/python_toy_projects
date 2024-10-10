import pygame
import time

#game of life
#0 is dead and 1 is alive
class cell:
    def __init__(self, s):
        self.state = s

pygame.init()
width, height = 1280, 720
cell_size = 6
grid_w, grid_h = int(width/cell_size), int(height/cell_size)
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("game of life")
rect_surface = pygame.Surface((width, height))
rect_surface.fill((20, 20, 20))

grid = [[cell(0) for _ in range(grid_h)] for _ in range(grid_w)]
grid_ = [[cell(0) for _ in range(grid_h)] for _ in range(grid_w)]

grid_color = (105, 255, 92)
brush_size = 3

def draw_mouse(rect_surface):
    mouse_pos = pygame.mouse.get_pos()
    for i in range(brush_size, grid_w-brush_size+1):
        for j in range(brush_size, grid_h-brush_size):
            rect = pygame.Rect(i * cell_size, j * cell_size, cell_size, cell_size)
            if (rect.collidepoint(mouse_pos)):
                for x in range(-brush_size, brush_size):
                    for y in range(-brush_size, brush_size):
                        rect = pygame.Rect((i+x) * cell_size, (j+y) * cell_size, cell_size, cell_size)
                        pygame.draw.rect(rect_surface, grid_color, rect)
                        if pygame.mouse.get_pressed()[0]:
                            grid[i+x][y+j].state = 1


iter=0
def simulate():
    global grid
    global grid_

    grid_ = [[cell(0) for _ in range(grid_h)] for _ in range(grid_w)]
    for j in range(grid_h):
        for i in range(grid_w):

            if(i == 0 or i == grid_w-1 or j == 0 or j == grid_h - 1 ):
                grid[i][j].state = 0
            else:
                an = 0
                for x in range(-1, 2, 1):
                    for y in range(-1, 2, 1):
                        if(grid[i+x][j+y].state == 1):
                            an+=1
                an-=grid[i][j].state
                if(grid[i][j].state == 1):
                    if(an < 2):
                        grid_[i][j].state = 0
                    if(an == 2 or an == 3):
                        grid_[i][j].state = 1
                    if(an>3):
                        grid_[i][j].state = 0
                else:
                    if(an == 3):
                        grid_[i][j].state =  1

    grid = grid_.copy()


def draw_grid():
    for i in range(grid_w):
        for j in range(grid_h):
            if(i==20):
                print("")
            if(grid_[i][j].state == 1):
                rect = pygame.Rect(i * cell_size, j*cell_size, cell_size, cell_size)
                pygame.draw.rect(rect_surface, grid_color, rect)


running = True

while running:
    iter+=1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:  # Q key pressed
        brush_size+=1
    if keys[pygame.K_e]:  # E key pressed
        brush_size-=1
    brush_size = max(brush_size, 0)

    rect_surface.fill((30, 30, 30))

    draw_grid()
    draw_mouse(rect_surface)
    window.blit(rect_surface, (0, 0))

    simulate()
    # Update the display
    pygame.display.flip()
# Quit pygame
pygame.quit()


