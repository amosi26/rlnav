import math
import random

import numpy as np
import pygame


pygame.init()

#width and height of the game window
WIDTH = 600
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple Navigation Game")

#colors 
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
BLACK = (0, 0, 0)


# robot settings
robot_size = 20
robot_x = 100
robot_y = 100
robot_speed = 5

# goal settings
goal_size = 20
goal_x = random.randint(0, WIDTH - goal_size)
goal_y = random.randint(0, HEIGHT - goal_size)

# font for showing distance on screen
font = pygame.font.SysFont(None, 36)

clock = pygame.time.Clock()

running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        robot_y -= robot_speed
    if keys[pygame.K_s]:
        robot_y += robot_speed
    if keys[pygame.K_a]:
        robot_x -= robot_speed
    if keys[pygame.K_d]:
        robot_x += robot_speed

    # keep robot inside screen
    robot_x = max(0, min(robot_x, WIDTH - robot_size))
    robot_y = max(0, min(robot_y, HEIGHT - robot_size))

    # collision with goal
    robot_rect = pygame.Rect(robot_x, robot_y, robot_size, robot_size)
    goal_rect = pygame.Rect(goal_x, goal_y, goal_size, goal_size)

    if robot_rect.colliderect(goal_rect):
        print("Goal reached!")
        goal_x = random.randint(0, WIDTH - goal_size)
        goal_y = random.randint(0, HEIGHT - goal_size)


     # center points
    robot_center_x = robot_x + robot_size / 2
    robot_center_y = robot_y + robot_size / 2
    goal_center_x = goal_x + goal_size / 2
    goal_center_y = goal_y + goal_size / 2

    # distance formula
    distance = math.sqrt((goal_center_x - robot_center_x) ** 2 + (goal_center_y - robot_center_y) ** 2)

    # turn distance into display text
    distance_text = font.render(f"Distance: {distance:.2f}", True, BLACK)


    # draw
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLUE, (robot_x, robot_y, robot_size, robot_size))
    pygame.draw.rect(screen, GREEN, (goal_x, goal_y, goal_size, goal_size))
    screen.blit(distance_text, (10, 10))
    pygame.display.flip()





pygame.quit()       