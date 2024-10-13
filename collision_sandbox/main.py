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

running = True

solver = VerletSolver(60, [], acenter=Vec2(WIDTH/2, HEIGHT/2), arad=HEIGHT/2)
timer = pygame.time.Clock()

def draw_objects(surf: pygame.Surface, objs: list[Object]):
    for o in objs:
        pygame.draw.circle(surf, color=WHITE, radius=o.radius, center=(o.pos.x, o.pos.y))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            solver.objects.append(Object(Vec2(mouse_pos[0], mouse_pos[1]), 15.0))

    draw_surf.fill(BG_COL)
    solver.update()
    pygame.draw.circle(draw_surf, radius=solver.area_radius, center=(WIDTH/2, HEIGHT/2), color=(0, 0, 0))
    draw_objects(draw_surf, solver.objects)
    window.blit(draw_surf, (0, 0))

    pygame.display.flip()
    timer.tick(60)
pygame.quit()