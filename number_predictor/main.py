import pygame
import torch
import torch.nn as nn
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the images
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

pygame.init()
width, height = 1280, 700
cell_size = int(height/28)

draw_surf = pygame.Surface((height, height))
ui_surf = pygame.Surface((width-height, height))
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("number predictor")
fill_col = (190, 190, 190)
bg_col = (30, 30, 30)
ui_bg_col = (200, 200, 200)
BLACK = (0, 0, 0)

font = pygame.font.Font(None, 36)

grid = [[0 for _ in range(28)] for _ in range(28)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()
model = model.to(device)

def draw_grid():
    for i in range(28):
        for j in range(28):
            rect = pygame.Rect(i * cell_size, j * cell_size, cell_size, cell_size)
            col = bg_col
            if(grid[i][j] == 1):
                col = fill_col
            pygame.draw.rect(draw_surf, col, rect)

def draw_mouse():
    mouse_pos = pygame.mouse.get_pos()
    for i in range(28):
        for j in range(28):
            rect = pygame.Rect(i * cell_size, j * cell_size, cell_size, cell_size)
            if(rect.collidepoint(mouse_pos)):
                pygame.draw.rect(draw_surf, fill_col, rect)
                if pygame.mouse.get_pressed()[0]:
                    grid[i][j] =1

def predict():
    img = [[0.0 for _ in range(28)] for _ in range(28)]
    for y in range(28):
        for x in range(28):
            img[y][x] = float(grid[x][y]) * 255.0#somehow only this works, maybe some dimension issue
    inp_arr = np.array(img)
    inp_tensor = torch.tensor(inp_arr, dtype=torch.float32)
    inp_tensor = inp_tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(inp_tensor)
        predicted_s = torch.softmax(output, 1)
        return predicted_s

def draw_text(surface, text, position, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)  # Render the text
    surface.blit(text_surface, position)  # Blit the text onto the surface

def draw_preds(preds):
    spacing = 60
    for x in range(10):
        rect = pygame.Rect(100, spacing*(x+1), preds[0][x].item() * 400, 30)
        pygame.draw.rect(ui_surf, BLACK, rect)
        draw_text(ui_surf, str(x), (80, spacing*(x+1)+2), BLACK)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            grid = [[0 for _ in range(28)] for _ in range(28)]

    draw_surf.fill(bg_col)
    draw_grid()
    draw_mouse()

    ui_surf.fill(ui_bg_col)
    preds = predict()
    draw_preds(preds)

    window.blit(draw_surf, (0, 0))
    window.blit(ui_surf, (height, 0))
    pygame.display.flip()

pygame.quit()



