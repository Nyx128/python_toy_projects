import pygame
from tree import quad_tree
from tree import BoundingBox

pygame.init()
width, height = 1280, 720
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("collision_sandbox")
BG_COL = (30, 30, 30)

draw_surf = pygame.Surface((width, height))

running = True

base = quad_tree(BoundingBox(0, 0, width, height), 10)
WHITE = (255, 255, 255)
CYAN = (0, 200, 255)

points=[]

def draw_quad_tree(qt: quad_tree, surf: pygame.Surface):
    drawn = False
    while(drawn != True):
        rect = pygame.Rect(qt.boundary.x, qt.boundary.y, qt.boundary.w, qt.boundary.h)
        pygame.draw.rect(surf, WHITE, rect, 1)
        if qt.divided:
            draw_quad_tree(qt.q1, surf)
            draw_quad_tree(qt.q2, surf)
            draw_quad_tree(qt.q3, surf)
            draw_quad_tree(qt.q4, surf)
            drawn = True
        else:
            drawn = True

def draw_points(surf: pygame.Surface):
    for p in points:
        pygame.draw.circle(surf, CYAN, center=p, radius=4.0)


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        mouse_pos= pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            points.append(mouse_pos)
            base.insert(mouse_pos[0], mouse_pos[1])

        keys=pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            print(base.get_neighbours(points[4][0], points[4][1]))

    draw_surf.fill(BG_COL)

    draw_points(draw_surf)
    draw_quad_tree(base, draw_surf)

    window.blit(draw_surf, (0, 0))

    pygame.display.flip()
pygame.quit()