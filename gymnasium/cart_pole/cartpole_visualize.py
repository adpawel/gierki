"""
Wizualizacja wytrenowanego agenta DQN w środowisku CartPole-v1.

Uruchomienie:
    python cartpole_visualize.py

Wymaga wytrenowanego modelu zapisanego jako 'best_policy_net.pt'
(zapisywanego automatycznie przez notatnik experiments.ipynb).

Sterowanie:
    R  - resetuj epizod
    Q  - wyjdź
"""

import math
import sys
import torch
import gymnasium as gym

try:
    import pygame
except ImportError:
    print("Brak pygame. Zainstaluj: pip install pygame")
    sys.exit(1)

from dqn_model import DQN_128x128, DQN_128x64

MODEL_PATH  = "best_policy_net.pt"
MODEL_CLASS = DQN_128x128   # zmień na DQN_128x64 jeśli zapisałeś tę konfigurację

SCREEN_W, SCREEN_H = 800, 400
FPS = 60

# Kolory
BG_COLOR      = (245, 245, 245)
TRACK_COLOR   = (80,  80,  80)
CART_COLOR    = (33, 150, 243)
CART_EDGE     = (21,  101, 192)
WHEEL_COLOR   = (50,  50,  50)
POLE_COLOR    = (229, 57,  53)
PIVOT_COLOR   = (255, 193,  7)
TEXT_COLOR    = (50,  50,  50)
GREEN_COLOR   = (46, 125,  50)
RED_COLOR     = (198, 40,  40)

SCALE     = 100
CART_Y    = SCREEN_H // 2 + 30
CART_W    = 60
CART_H    = 30
WHEEL_R   = 8
POLE_LEN  = 120
POLE_W    = 8

def load_model(path, model_class):
    device = torch.device("cpu")
    env_tmp = gym.make("CartPole-v1")
    n_obs = env_tmp.observation_space.shape[0]
    n_act = env_tmp.action_space.n
    env_tmp.close()

    model = model_class(n_obs, n_act).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device


def draw_frame(screen, font, font_large, obs, step, total_reward, done):
    cart_pos, _, pole_angle, _ = obs

    screen.fill(BG_COLOR)

    pygame.draw.line(screen, TRACK_COLOR,
                     (0, CART_Y + CART_H // 2 + WHEEL_R * 2),
                     (SCREEN_W, CART_Y + CART_H // 2 + WHEEL_R * 2), 3)

    cx = int(SCREEN_W // 2 + cart_pos * SCALE)
    cy = CART_Y

    # Wózek
    cart_rect = pygame.Rect(cx - CART_W // 2, cy - CART_H // 2, CART_W, CART_H)
    pygame.draw.rect(screen, CART_COLOR, cart_rect, border_radius=6)
    pygame.draw.rect(screen, CART_EDGE,  cart_rect, width=2, border_radius=6)

    # Koła
    for dx in [-CART_W // 2 + 10, CART_W // 2 - 10]:
        pygame.draw.circle(screen, WHEEL_COLOR,
                           (cx + dx, cy + CART_H // 2 + WHEEL_R), WHEEL_R)

    # Kij
    pivot_x = cx
    pivot_y = cy - CART_H // 2
    tip_x   = int(pivot_x + POLE_LEN * math.sin(pole_angle))
    tip_y   = int(pivot_y - POLE_LEN * math.cos(pole_angle))

    pygame.draw.line(screen, POLE_COLOR, (pivot_x, pivot_y), (tip_x, tip_y), POLE_W)

    pygame.draw.circle(screen, PIVOT_COLOR, (pivot_x, pivot_y), 7)

    angle_deg   = math.degrees(pole_angle)
    angle_color = RED_COLOR if abs(angle_deg) > 8 else GREEN_COLOR

    texts = [
        (font_large, f"Krok: {step}   Wynik: {total_reward:.0f}", TEXT_COLOR,  20, 20),
        (font,       f"Kąt kija: {angle_deg:+.1f}°",              angle_color, 20, 55),
        (font,       f"Pozycja wózka: {cart_pos:+.3f}",           TEXT_COLOR,  20, 75),
        (font,       "R – reset   Q – wyjdź",                     TEXT_COLOR,  20, SCREEN_H - 30),
    ]
    for f, txt, col, x, y in texts:
        screen.blit(f.render(txt, True, col), (x, y))

    if done:
        msg = font_large.render("Epizod zakończony! Naciśnij R aby powtórzyć.", True, RED_COLOR)
        screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, SCREEN_H // 2 - 20))

    pygame.display.flip()



def main():
    print(f"Ładowanie modelu z: {MODEL_PATH}")
    try:
        model, device = load_model(MODEL_PATH, MODEL_CLASS)
    except FileNotFoundError:
        print(f"Nie znaleziono pliku '{MODEL_PATH}'.")
        print("Uruchom najpierw notatnik experiments.ipynb, który zapisze wytrenowany model.")
        sys.exit(1)

    env = gym.make("CartPole-v1")

    pygame.init()
    screen    = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("CartPole - wytrenowany agent DQN")
    clock     = pygame.time.Clock()
    font      = pygame.font.SysFont("monospace", 18)
    font_large = pygame.font.SysFont("monospace", 22, bold=True)

    obs, _       = env.reset()
    step         = 0
    total_reward = 0.0
    done         = False

    print("Wizualizacja uruchomiona. R - reset, Q - wyjdź.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); env.close(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit(); env.close(); sys.exit()
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
                    step, total_reward, done = 0, 0.0, False

        draw_frame(screen, font, font_large, obs, step, total_reward, done)

        if not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = model(state_t).max(1).indices.item()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step         += 1
            done          = terminated or truncated

        clock.tick(FPS)


if __name__ == "__main__":
    main()
