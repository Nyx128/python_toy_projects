import pygame
import torch
import torch.nn as nn
import numpy as np

    
class CNN_V0(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


pygame.init()
width, height = 1280, 700
cell_size = int(height/28)

draw_surf = pygame.Surface((height, height))
ui_surf = pygame.Surface((width-height, height))
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("number predictor")
fill_col = (255, 255, 255)
bg_col = (30, 30, 30)
ui_bg_col = (200, 200, 200)
BLACK = (0, 0, 0)

font = pygame.font.Font(None, 36)

grid = [[0 for _ in range(28)] for _ in range(28)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_V0(input_shape=1, hidden_units=10, output_shape=10)
model.load_state_dict(torch.load('number_cnn.pth'))
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
            img[y][x] = float(grid[x][y]) * 255.0#i messed up dimensions it was [row][col] and not [col][row] so i switch it
    inp_arr = np.array(img)
    inp_tensor = torch.tensor(inp_arr, dtype=torch.float32)
    inp_tensor = inp_tensor.reshape([1, 1, 28, 28])
    #print(inp_tensor.shape)

    with torch.no_grad():
        output = model(inp_tensor)
        predicted_s = torch.softmax(output, 1)
        return predicted_s

def draw_text(surface, text, position, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)  # Render the text
    surface.blit(text_surface, position)

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



