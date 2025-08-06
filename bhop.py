import pygame
from sys import exit
import pygame.draw_py

forward=[1,0,0]
right=[0,-1,0]

def air_move(fmove,smove):
    
    pass

pygame.init()
screen=pygame.display.set_mode((1000,1000))
pygame.display.set_caption("bhop simulation")
clock=pygame.time.Clock()
player_x=200
player_y=200
player_surface=pygame.image.load('player.png').convert_alpha()
player_rect=player_surface.get_rect(center=(player_x,player_y))
bg_surface=pygame.Surface((1000,1000))

while True:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            exit()
    keys=pygame.key.get_pressed()
    if keys[pygame.K_w]:
        player_y-=1
    if keys[pygame.K_s]:
        player_y+=1
    if keys[pygame.K_a]:
        player_x-=1
    if keys[pygame.K_d]:
        player_x+=1
    player_rect=player_surface.get_rect(center=(player_x,player_y))
    screen.blit(bg_surface,(0,0))
    screen.blit(player_surface,player_rect)
    pygame.display.update()
    clock.tick(60)
