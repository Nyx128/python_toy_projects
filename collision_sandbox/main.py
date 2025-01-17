import pygame
from verlet import *

pygame.init()
WIDTH, HEIGHT = 1000, 1000
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("collision_sandbox")

draw_surf = pygame.Surface((WIDTH, HEIGHT))
BG_COL = (25, 25, 25)
WHITE = (235, 235, 235)
draw_surf.fill(BG_COL)

base = quad_tree(BoundingBox(WIDTH/2 - HEIGHT/2, 0, HEIGHT, HEIGHT), 1)

running = True

solver = VerletSolver(60, [], acenter=Vec2(WIDTH/2, HEIGHT/2), arad=HEIGHT/2)
timer = pygame.time.Clock()

def draw_objects(surf: pygame.Surface, objs: list[Object]):
    for o in objs:
        pygame.draw.circle(surf, color=WHITE, radius=o.radius, center=(o.pos.x, o.pos.y))

def sample_mouse():
    if pygame.mouse.get_pressed()[0]:
        mouse_pos =pygame.mouse.get_pos()
        obj = Object(Vec2(mouse_pos[0], mouse_pos[1]), 15.0)
        obj.setVelocity(Vec2(0, 100), 1.0/60)
        solver.objects.append(obj)

def draw_quad_tree(qt: quad_tree, surf: pygame.Surface):
    if qt.divided:
        draw_quad_tree(qt.q1, surf)
        draw_quad_tree(qt.q2, surf)
        draw_quad_tree(qt.q3, surf)
        draw_quad_tree(qt.q4, surf)
    else:
        rect = pygame.Rect(qt.boundary.x, qt.boundary.y, qt.boundary.w, qt.boundary.h)
        pygame.draw.rect(surf, color = WHITE, rect= rect, width=1 )



while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    draw_surf.fill(BG_COL)
    sample_mouse()
    base = quad_tree(BoundingBox(WIDTH / 2 - HEIGHT / 2, 0, HEIGHT, HEIGHT), 30)
    for x in solver.objects:
        base.insert(x)
    solver.update(base)

    pygame.draw.circle(draw_surf, radius=solver.area_radius, center=(WIDTH/2, HEIGHT/2), color=(0, 0, 0))
    draw_objects(draw_surf, solver.objects)
    window.blit(draw_surf, (0, 0))

    pygame.display.flip()
    timer.tick(60)
pygame.quit()