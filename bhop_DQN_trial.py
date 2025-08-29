import math
import pygame
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MOUSE_SENSITIVITY = 0.5
MAX_SPEED = 400
ACCELERATE = 14
AIR_ACCELERATE = 2
FRICTION = 10
GRAVITY = 1000
AIR_SPEED_CAP = 60
MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(7, 256, 6)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, player, game):
        yaw = ((player.yaw + math.pi) % (2*math.pi)) - math.pi
        yaw /= math.pi
        pos_x, pos_y = player.pos.x/1000.0, player.pos.y/1000.0
        vel_x, vel_y = player.velocity.x/MAX_SPEED, player.velocity.y/MAX_SPEED
        z_pos = player.z_pos/40.0   # pick your max jump height
        distance = player.pos.distance_to(game.end_pos.copy()) / math.hypot(1000,1000)
        return np.array([yaw, pos_x, pos_y, vel_x, vel_y, z_pos, distance], dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: exploration
        self.epsilon = max(5, 80 - self.n_games)

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)

        if random.randint(0, 200) < self.epsilon:
            # explore randomly
            mx = random.uniform(-1, 1)   # random mouse movement
            buttons = [random.randint(0,1) for _ in range(5)]  # random booleans
            action = [mx] + buttons
        else:
            # exploit: use model prediction
            mx = prediction[0].item()  # first output = mouse delta
            buttons = [1 if p > 0 else 0 for p in prediction[1:].tolist()]
            action = [mx] + buttons

        return action

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

    def handle_input(self,mx,up_press,down_press,left_press,right_press,jump):
        self.yaw += mx * MOUSE_SENSITIVITY

        jump_pressed = False
        fmove = 0
        smove = 0

        if up_press:
            fmove += 1
        if down_press:
            fmove -= 1
        if right_press:
            smove += 1
        if left_press:
            smove -= 1
        if jump:
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

    def update(self, dt,action):
        mx, up, down, left, right, jump = action
        wishdir, wishspeed, jump_pressed = self.handle_input(mx,up,down,left,right,jump)
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
            player.pos = self.start_pos.copy() #self.start_pos wasn't working
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
agent=Agent()
while running:
    dt_ms = clock.tick(60)
    dt = dt_ms / 1000.0
    total_time += dt_ms

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    # Agent observes
    state_old = agent.get_state(player, game)

    # Agent acts
    action = agent.get_action(state_old)

    # Optional: clamp/scale mx to something sane
    RANGE_MX = 20.0
    action[0] = max(-RANGE_MX, min(RANGE_MX, action[0]))

    # Step the game with the action
    player.update(dt, action)
    # Check finish
    finished = game.check_finish(player, total_time)

    # Build next state
    state_new = agent.get_state(player, game)

    # Compute reward (simple shaping example)
    prev_dist = state_old[-1]
    new_dist = state_new[-1]
    reward = 0.0
    reward += 50.0 if finished else 0.0
    reward += (prev_dist - new_dist) * 10.0        # per-step progress reward
    #reward -= 0.01                                 # time penalty

    # TIMEOUT CHECK
    time_since_finish = (total_time - game.last_finish) / 1000.0
    done = False
    if finished:
        done = True
    elif time_since_finish >= 15:
        reward -= 5.0   # punish for taking too long
        done = True
        game.last_finish = total_time  # reset the timer so next run starts fresh
        player.pos = game.start_pos.copy() 
        player.velocity = pygame.math.Vector2(0, 0)  # stop sliding
        player.z_pos = 0
        player.z_velocity = 0 # respawn at start

    if done:
        agent.n_games += 1
        if agent.n_games % 10 == 0:
            agent.model.save(f"model_ep{agent.n_games}.pth")
        print(f"EP {agent.n_games} reward={reward:.2f}")

    if done and len(agent.memory)>=BATCH_SIZE:
        agent.train_long_memory()
    # Train & remember
    agent.train_short_memory(state_old, action, reward, state_new, done)
    agent.remember(state_old, action, reward, state_new, done)
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
        f"EPISODE: {agent.n_games}",
        f"pos: ({player.pos.x:.2f}, {player.pos.y:.2f})",
        f"vel: ({player.velocity.x:.2f}, {player.velocity.y:.2f})",
        f"grounded: {player.on_ground}",
        f"z_pos: {player.z_pos:.2f}",
        f"reward:{reward:.6f}"
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
