import gymnasium as gym
import math
import pygame
from gymnasium import spaces
import numpy as np


MOUSE_SENSITIVITY = 0.5
MAX_SPEED = 400
ACCELERATE = 14
AIR_ACCELERATE = 2
FRICTION = 10
GRAVITY = 1000
AIR_SPEED_CAP = 60

class Player:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.velocity = pygame.math.Vector2(0, 0)
        self.z_pos = 0
        self.z_velocity = 0
        self.yaw = 0.0
        self.on_ground = True
        self.alpha=0

def update_opacity(player):
    min_z, max_z = -40, 0
    normalized_z = max(0, min(1, (player.z_pos - min_z) / (max_z - min_z)))
    alpha = normalized_z * 255
    player.alpha=int(alpha)

def handle_input(player, mx, up_press, down_press, left_press, right_press, jump):
    player.yaw += mx * MOUSE_SENSITIVITY

    fmove = int(up_press) - int(down_press)
    smove = int(right_press) - int(left_press)
    jump_pressed = bool(jump)

    forward = pygame.math.Vector2(math.cos(player.yaw), math.sin(player.yaw))
    right   = pygame.math.Vector2(-math.sin(player.yaw), math.cos(player.yaw))

    wishvel = forward * fmove + right * smove
    if wishvel.length_squared() > 0:
        wishspeed = wishvel.length() * MAX_SPEED
        wishdir = wishvel.normalize()
    else:
        wishspeed = 0
        wishdir = pygame.math.Vector2(0, 0)

    return wishdir, wishspeed, jump_pressed

def friction(player, dt):
    speed = player.velocity.length()
    if speed < 0.5:
        player.velocity.x = 0
        player.velocity.y = 0
        return
    drop = speed * FRICTION * dt
    new_speed = max(speed - drop, 0)
    player.velocity *= new_speed / speed

def accelerate(player, wishdir, wishspeed, accel, dt):
    currentspeed = player.velocity.dot(wishdir)
    addspeed = wishspeed - currentspeed
    if addspeed <= 0:
        return
    accelspeed = accel * wishspeed * dt
    if accelspeed > addspeed:
        accelspeed = addspeed
    player.velocity += wishdir * accelspeed

def airaccelerate(player, wishdir, wishspeed, accel, dt):
    if wishspeed > AIR_SPEED_CAP:
        wishspeed = AIR_SPEED_CAP
    currentspeed = player.velocity.dot(wishdir)
    addspeed = wishspeed - currentspeed
    if addspeed <= 0:
        return
    accelspeed = accel * wishspeed * dt
    if accelspeed > addspeed:
        accelspeed = addspeed
    player.velocity += wishdir * accelspeed

def do_jump(player, jump_pressed):
    if jump_pressed and player.on_ground:
        player.z_velocity = -250
        player.on_ground = False

def handle_jump(player, dt):
    player.z_velocity += GRAVITY * dt
    player.z_pos += player.z_velocity * dt
    if player.z_pos >= 0:
        player.z_pos = 0
        player.z_velocity = 0
        player.on_ground = True


class BhopEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 1/60.0
        self.player = None
        self.start_pos = pygame.math.Vector2(500, 50)
        self.end_pos = pygame.math.Vector2(500, 950)
        # Action space: [mx, up, down, left, right, jump]
        self.action_space = spaces.Box(
            low=np.array([-20, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([20, 1, 1, 1, 1, 1], dtype=np.float32)
        )
        # Observation: [x, y, z, vx, vy, yaw]
        low = np.array([0, 0, -40, -500, -500, -180], dtype=np.float32)
        high = np.array([1000, 1000, 200, 500, 500, 180], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        mx, up, down, left, right, jump = action

        up = 1 if up > 0.5 else 0
        down = 1 if down > 0.5 else 0
        left = 1 if left > 0.5 else 0
        right = 1 if right > 0.5 else 0
        jump = 1 if jump > 0.5 else 0

        wishdir, wishspeed, jump_pressed = handle_input(
            self.player, mx, up, down, left, right, jump
        )

        self.player.on_ground = (self.player.z_pos == 0)

        if self.player.on_ground:
            friction(self.player, self.dt)
            do_jump(self.player, jump_pressed)
            accelerate(self.player, wishdir, wishspeed, ACCELERATE, self.dt)
        else:
            airaccelerate(self.player, wishdir, wishspeed, AIR_ACCELERATE, self.dt)

        handle_jump(self.player, self.dt)
        self.player.pos += self.player.velocity * self.dt
        update_opacity(self.player)

        # Clamp to map
        self.player.pos.x = np.clip(self.player.pos.x, 0, 1000)
        self.player.pos.y = np.clip(self.player.pos.y, 0, 1000)

        # --- Reward shaping ---
        if not hasattr(self, "prev_pos"):
            self.prev_pos = self.player.pos.copy()

        prev_dist = np.linalg.norm(self.prev_pos - self.end_pos)
        new_dist = np.linalg.norm(self.player.pos - self.end_pos)

        reward = (prev_dist - new_dist) * 0.1 - 0.01  # closer → reward, time penalty

        # Update prev_pos for next step
        self.prev_pos = self.player.pos.copy()

        # --- Termination ---
        done = False
        if new_dist < 30:   # reached goal
            reward += 50.0
            done = True

        obs = self._get_obs()
        info = {}

        return obs, reward, done, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player = Player((500, 50))
        self.player.velocity = pygame.math.Vector2(0, 0)
        self.player.z_pos = 0
        self.player.z_velocity = 0
        self.player.yaw = 0.0
        obs = self._get_obs()
        return obs, {}

    def render(self, mode="human"):
        if not hasattr(self, "screen"):
            pygame.init()
            self.font = pygame.font.SysFont("Ubuntu Sans Mono", 24)
            self.screen = pygame.display.set_mode((1000, 1000))
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            self.bg_surface = pygame.Surface((1000, 1000))
            self.start_img = pygame.image.load("start.png").convert_alpha()
            self.end_img = pygame.image.load("end.png").convert_alpha()
            self.player_img = pygame.image.load("player.png").convert_alpha()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill((0, 0, 0))
        if getattr(self, "_closed", False):
            return

        # Rotate player sprite by yaw
        angle_degrees = -math.degrees(self.player.yaw) - 90
        rotated_surface = pygame.transform.rotate(self.player_img, angle_degrees)
        rotated_surface.set_alpha(self.player.alpha)
        rotated_rect = rotated_surface.get_rect(center=self.player.pos)

        # Draw background + objects
        self.screen.blit(self.bg_surface, (0, 0))
        self.screen.blit(self.start_img, (self.start_pos.x - 50, self.start_pos.y - 50))
        self.screen.blit(self.end_img, (self.end_pos.x - 50, self.end_pos.y - 50))
        self.screen.blit(rotated_surface, rotated_rect)

        # Draw forward direction line
        dir_x = self.player.pos.x + math.cos(self.player.yaw) * 50
        dir_y = self.player.pos.y + math.sin(self.player.yaw) * 50
        pygame.draw.circle(self.screen, (128, 128, 128), (int(dir_x), int(dir_y)), 5)

        # HUD info
        info_lines = [
            f"pos: ({self.player.pos.x:.2f}, {self.player.pos.y:.2f})",
            f"vel: ({self.player.velocity.x:.2f}, {self.player.velocity.y:.2f})",
            f"grounded: {self.player.on_ground}",
            f"z_pos: {self.player.z_pos:.2f}",
        ]
        y_offset = 10
        for line in info_lines:
            text_surface = self.font.render(line, True, (255, 255, 255))
            text_rect = text_surface.get_rect(topright=(990, y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += 20

        pygame.display.update()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2)
            )
    def close(self):
        if hasattr(self, "screen"):
            pygame.display.quit()
        pygame.quit()

    def _get_obs(self):
        return np.array([
            self.player.pos.x,
            self.player.pos.y,
            self.player.z_pos,
            self.player.velocity.x,
            self.player.velocity.y,
            math.degrees(self.player.yaw) % 360 - 180
        ], dtype=np.float32)
