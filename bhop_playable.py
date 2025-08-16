import math
import pygame

MOUSE_SENSITIVITY = 0.002
MAX_SPEED = 200
ACCELERATE = 7
AIR_ACCELERATE = 1
FRICTION = 10
GRAVITY = 900
AIR_SPEED_CAP = 30

class Player:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.velocity = pygame.math.Vector2(0, 0)
        self.z_pos = 0
        self.z_velocity = 0
        self.yaw = 0.0
        self.player_surface = pygame.image.load('player.png').convert_alpha()
        self.on_ground = True

    def update_opacity(self):
        min_z, max_z = -40, 0
        normalized_z = max(0, min(1, (self.z_pos - min_z) / (max_z - min_z)))
        alpha = normalized_z * 255
        self.player_surface.set_alpha(min(int(alpha), 255))

    def handle_input(self):
        mx, my = pygame.mouse.get_rel()
        self.yaw += mx * MOUSE_SENSITIVITY

        jump_pressed = False
        fmove = 0
        smove = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            fmove += 1
        if keys[pygame.K_DOWN]:
            fmove -= 1
        if keys[pygame.K_RIGHT]:
            smove += 1
        if keys[pygame.K_LEFT]:
            smove -= 1
        if keys[pygame.K_RSHIFT]:
            jump_pressed = True

        forward = pygame.math.Vector2(math.cos(self.yaw), math.sin(self.yaw))
        right   = pygame.math.Vector2(-math.sin(self.yaw), math.cos(self.yaw))

        wishvel = forward * fmove + right * smove
        if wishvel.length_squared() > 0:
            wishspeed = wishvel.length() * MAX_SPEED
            wishdir = wishvel.normalize()
        else:
            wishspeed = 0
            wishdir = pygame.math.Vector2(0, 0)

        return wishdir, wishspeed, jump_pressed

    def pm_friction(self, dt):
        speed = self.velocity.length()
        if speed < 0.5:
            self.velocity.x = 0
            self.velocity.y = 0
            return
        drop = speed * FRICTION * dt
        new_speed = max(speed - drop, 0)
        self.velocity *= new_speed / speed

    def accelerate(self, wishdir, wishspeed, accel, dt):
        currentspeed = self.velocity.dot(wishdir)
        addspeed = wishspeed - currentspeed
        if addspeed <= 0:
            return
        accelspeed = accel * wishspeed * dt
        if accelspeed > addspeed:
            accelspeed = addspeed
        self.velocity += wishdir * accelspeed

    def airaccelerate(self, wishdir, wishspeed, accel, dt):
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

    def jump(self, jump_pressed):
        if jump_pressed and self.on_ground:
            self.z_velocity = -250
            self.on_ground = False

    def handle_jump(self, dt):
        self.z_velocity += GRAVITY * dt
        self.z_pos += self.z_velocity * dt
        if self.z_pos >= 0:
            self.z_pos = 0
            self.z_velocity = 0
            self.on_ground = True

    def update(self, dt):
        wishdir, wishspeed, jump_pressed = self.handle_input()
        if self.z_pos == 0:
            self.on_ground = True
        else:
            self.on_ground = False

        if self.on_ground:
            self.pm_friction(dt)
            self.jump(jump_pressed)
            self.accelerate(wishdir, wishspeed, ACCELERATE, dt)
        else:
            self.airaccelerate(wishdir, wishspeed, AIR_ACCELERATE, dt)

        self.handle_jump(dt)
        self.pos += self.velocity * dt

        half_w = self.player_surface.get_width() / 2
        half_h = self.player_surface.get_height() / 2
        self.pos.x = max(half_w, min(self.pos.x, 1000 - half_w))
        self.pos.y = max(half_h, min(self.pos.y, 1000 - half_h))
        self.update_opacity()

class Game:
    def __init__(self):
        self.start_pos = pygame.math.Vector2(500, 50)
        self.end_pos = pygame.math.Vector2(500, 950)
        self.last_finish = 0

    def check_finish(self, player, current_time_ms):
        if player.pos.distance_to(self.end_pos) <= 100:  # 100px radius check
            player.pos = pygame.math.Vector2(500,50) #self.start_pos wasn't working
            self.last_finish = current_time_ms
            return True
        return False

pygame.init()
font = pygame.font.SysFont("Ubuntu Sans Mono", 24)
screen = pygame.display.set_mode((1000, 1000))
clock = pygame.time.Clock()
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

bg_surface = pygame.Surface((1000, 1000))
game = Game()
player = Player(game.start_pos)

start = pygame.Surface((100, 100))
start=pygame.image.load("start.png").convert_alpha()

end = pygame.Surface((100, 100),)
end=pygame.image.load("end.png").convert_alpha()


total_time = 0
running = True

while running:
    dt_ms = clock.tick(60)
    dt = dt_ms / 1000.0
    total_time += dt_ms

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    player.update(dt)
    game.check_finish(player, total_time)

    screen.fill((0, 0, 0))

    angle_degrees = -math.degrees(player.yaw) - 90
    rotated_surface = pygame.transform.rotate(player.player_surface, angle_degrees)
    rotated_rect = rotated_surface.get_rect(center=player.pos)
    screen.blit(bg_surface, (0, 0))

    screen.blit(start, (game.start_pos.x - 50, game.start_pos.y - 50))
    screen.blit(end, (game.end_pos.x - 50, game.end_pos.y - 50))

    screen.blit(rotated_surface, rotated_rect)

    dir_x = player.pos.x + math.cos(player.yaw) * 50
    dir_y = player.pos.y + math.sin(player.yaw) * 50
    pygame.draw.circle(screen, (128, 128, 128), (int(dir_x), int(dir_y)), 5)

    info_lines = [
        f"pos: ({player.pos.x:.2f}, {player.pos.y:.2f})",
        f"vel: ({player.velocity.x:.2f}, {player.velocity.y:.2f})",
        f"grounded: {player.on_ground}",
        f"z_pos: {player.z_pos:.2f}"
    ]
    y_offset = 10
    for line in info_lines:
        text_surface = font.render(line, True, (255, 255, 255))
        text_rect = text_surface.get_rect(topright=(990, y_offset))
        screen.blit(text_surface, text_rect)
        y_offset += 20

    time_since_finish = (total_time - game.last_finish) / 1000.0
    timer_text = font.render(f"{time_since_finish:.2f}s", True, (255, 255, 255))
    timer_rect = timer_text.get_rect(center=(50,50 ))
    screen.blit(timer_text, timer_rect)

    pygame.display.update()

pygame.quit()
