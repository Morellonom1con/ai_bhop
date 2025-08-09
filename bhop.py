import pygame
from sys import exit

fmove=0
smove=0

maxspeed=10
onground=1

forward=[1,0,0]
right=[0,-1,0]

wishvel=[0,0]
wishdir=[0,0]

def vec_scale(in_vec,scale,out_vec):
    out_vec[0]=in_vec[0]*scale
    out_vec[1]=in_vec[1]*scale

def vec_normalize(v):
    ilength=0
    length=v[0]*v[0]+v[1]*v[1]
    length=length**0.5
    if length:
        ilength=1/length
        v[0]*=ilength
        v[1]*=ilength
    return length

def catagorizePosition(player_x,player_y,player_z,terrain):
    if player_z==terrain[player_x][player_y]:
        onground=1
    else:
        onground=-1

def air_move():
    for i in range(2):
        wishvel[i]=forward[i]*fmove+right[i]*smove
    wishdir=wishvel
    wishspeed=vec_normalize(wishdir)

    if wishspeed>maxspeed:
        vec_scale(wishvel,maxspeed/wishspeed,wishvel)
        wishspeed=maxspeed

    if onground!=-1:
        pass
    pass

def accelerate():
    pass

def playermove():
    pass


pygame.init()
screen=pygame.display.set_mode((1000,1000))
pygame.display.set_caption("bhop simulation")
clock=pygame.time.Clock()
player_x=500
player_y=500
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
