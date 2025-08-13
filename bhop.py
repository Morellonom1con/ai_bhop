import math
import pygame

MOUSE_SENSITIVITY = 0.002
MAX_SPEED = 400
ACCELERATE = 10
AIR_ACCELERATE = 1
FRICTION = 6
GRAVITY = 800  
AIR_SPEED_CAP = 30

class Player:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.velocity = pygame.math.Vector2(0, 0)
        self.yaw = 0.0
        self.player_surface=pygame.image.load('player.png').convert_alpha()
        self.on_ground = True

    def handle_input(self):
        mx, my = pygame.mouse.get_rel()
        self.yaw += mx * MOUSE_SENSITIVITY

        fmove = 0
        smove = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            fmove += 1
        if keys[pygame.K_s]:
            fmove -= 1
        if keys[pygame.K_d]:
            smove += 1
        if keys[pygame.K_a]:
            smove -= 1

        forward = pygame.math.Vector2(math.cos(self.yaw), math.sin(self.yaw))
        right   = pygame.math.Vector2(-math.sin(self.yaw), math.cos(self.yaw))
        
        wishvel = forward * fmove + right * smove
        if wishvel.length_squared() > 0:
            wishspeed = wishvel.length() * MAX_SPEED
            wishdir = wishvel.normalize()
        else:
            wishspeed = 0
            wishdir = pygame.math.Vector2(0, 0)

        return wishdir, wishspeed

    def pm_friction(self, dt):
        speed = self.velocity.length()
        if speed < 0.1:
            return
        drop = speed * FRICTION * dt
        new_speed = max(speed - drop, 0)
        self.velocity *= new_speed / speed

    def pm_accelerate(self, wishdir, wishspeed, accel, dt):
        currentspeed = self.velocity.dot(wishdir)
        addspeed = wishspeed - currentspeed
        if addspeed <= 0:
            return
        accelspeed = accel * wishspeed * dt
        if accelspeed > addspeed:
            accelspeed = addspeed
        self.velocity += wishdir * accelspeed

    def pm_airaccelerate(self, wishdir, wishspeed, accel, dt):
        if wishspeed > AIR_SPEED_CAP:
            wishspeed = AIR_SPEED_CAP
        currentspeed = self.velocity.dot(wishdir)
        addspeed = wishspeed - currentspeed
        if addspeed <= 0:
            return
        accelspeed = accel * wishspeed * dt
        if accelspeed > addspeed:
            accelspeed = addspeed
        self.velocity += wishdir * accelspeed

    def update(self, dt):
        wishdir, wishspeed = self.handle_input()

        if self.on_ground:
            self.pm_friction(dt)
            self.pm_accelerate(wishdir, wishspeed, ACCELERATE, dt)
        else:
            self.pm_airaccelerate(wishdir, wishspeed, AIR_ACCELERATE, dt)
            self.velocity.y += GRAVITY * dt  # simple gravity

        self.pos += self.velocity * dt

pygame.init()
font = pygame.font.SysFont(name= "Ubuntu Sans Mono",size= 24)
screen = pygame.display.set_mode((1000, 1000))
clock = pygame.time.Clock()
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)
bg_surface=pygame.Surface((1000,1000))
player = Player((400, 300))

running = True
while running:
    dt = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    player.update(dt)

    screen.fill((0, 0, 0))
    angle_degrees = -math.degrees(player.yaw) - 90 
    rotated_surface = pygame.transform.rotate(player.player_surface, angle_degrees)
    rotated_rect = rotated_surface.get_rect(center=player.pos)

    screen.blit(bg_surface, (0, 0))
    screen.blit(rotated_surface, rotated_rect)
    info_lines = [
            f"pos: ({player.pos.x:.2f}, {player.pos.y:.2f})",
            f"vel: ({player.velocity.x:.2f}, {player.velocity.y:.2f})",
            f"grounded: {player.on_ground}",
            f"alpha: {player.player_surface.get_alpha()}"
        ]

    y_offset = 10
    for line in info_lines:
        text_surface = font.render(line, True, (255, 255, 255))
        text_rect = text_surface.get_rect(topright=(990, y_offset))
        screen.blit(text_surface, text_rect)
        y_offset += 20
    pygame.display.update()

pygame.quit()
