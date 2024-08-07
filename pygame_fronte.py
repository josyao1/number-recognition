import pygame
import sys
from pygame.locals import K_ESCAPE, K_r
import requests
import io

# Initialize Pygame
pygame.init()

# Set up the window and drawing surface
screen = pygame.display.set_mode((400, 400))
drawSurface = pygame.Surface((400, 400))
pygame.display.set_caption('Digit Recognizer')

# Set the background to white
drawSurface.fill((255, 255, 255))

def save_image(surface, path):
    pygame.image.save(surface, path)

def send_image_for_prediction(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post('http://127.0.0.1:5000/predict', files={'file': f})
        prediction = response.json().get('prediction')
        return prediction

# Main loop
running = True
drawing = False
last_pos = None
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            if event.key == K_r:
                drawSurface.fill((255, 255, 255))  # Clear the drawing surface
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
                last_pos = event.pos
            if event.button == 3:
                save_image(drawSurface, 'drawn_digit.png')
                prediction = send_image_for_prediction('drawn_digit.png')
                print(f"Predicted digit: {prediction}")
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        if event.type == pygame.MOUSEMOTION and drawing:
            pygame.draw.line(drawSurface, (0, 0, 0), last_pos, event.pos, 10)  # Draw with black color
            last_pos = event.pos

    screen.fill((255, 255, 255))  # White background for the screen
    screen.blit(drawSurface, (0, 0))
    pygame.display.flip()

pygame.quit()
