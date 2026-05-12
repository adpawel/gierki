import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame

class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # --- PRZESTRZEŃ AKCJI (Dyskretna) ---
        # 0: przyspiesz, 1: hamuj/cofaj, 2: skręć w lewo, 3: skręć w prawo
        self.action_space = spaces.Discrete(4)

        # --- PRZESTRZEŃ OBSERWACJI (Ciągła - Box) ---
        # Wszystkie wartości znormalizowane do zakresu [-1, 1] lub [0, 1]
        # Dzięki temu sieć neuronowa PPO uczy się znacznie efektywniej!
        # [0] car_x/800, [1] car_y/600, [2] car_angle/180, [3] car_speed/5,
        # [4] target_x/800, [5] target_y/600, [6] target_angle/180,
        # [7] obs1_x/800, [8] obs1_y/600, [9] obs2_x/800, [10] obs2_y/600,
        # [11] dx do celu (znorm.), [12] dy do celu (znorm.), [13] dystans (znorm.)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32
        )

        # Zapisujemy poprzedni dystans do obliczenia nagrody za postęp
        self.prev_dist = None

    def _get_observation(self):
        """Buduje znormalizowany wektor obserwacji."""
        dist = np.sqrt((self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2)
        # Normalizujemy dystans przez przekątną ekranu (~1000px)
        norm_dist = np.clip(dist / 1000.0, 0.0, 1.0)
        # Znormalizowane różnice kierunkowe
        dx = np.clip((self.target_x - self.car_x) / 800.0, -1.0, 1.0)
        dy = np.clip((self.target_y - self.car_y) / 600.0, -1.0, 1.0)

        obs = np.array([
            self.car_x / 800.0,
            self.car_y / 600.0,
            (self.car_angle % 360) / 180.0 - 1.0,  # zakres [-1, 1]
            self.car_speed / 5.0,
            self.target_x / 800.0,
            self.target_y / 600.0,
            self.target_angle / 180.0,
            self.obs1_x / 800.0,
            self.obs1_y / 600.0,
            self.obs2_x / 800.0,
            self.obs2_y / 600.0,
            dx,
            dy,
            norm_dist,
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Ustawienie samochodu na pozycji startowej
        self.car_x = 400.0 + random.uniform(-30, 30)
        self.car_y = 500.0 + random.uniform(-20, 20)
        self.car_angle = random.uniform(-10, 10)  # lekki losowy kąt
        self.car_speed = 0.0

        # Losowanie typu miejsca: 0 - równoległe, 1 - prostopadłe
        parking_type = random.choice([0, 1])

        if parking_type == 0:
            # RÓWNOLEGŁE (wzdłuż osi X)
            self.target_angle = 0.0
            self.target_x = 400.0
            self.target_y = 100.0
            self.obs1_x = self.target_x + 140.0
            self.obs1_y = self.target_y
            self.obs2_x = self.target_x - 140.0
            self.obs2_y = self.target_y
        else:
            # PROSTOPADŁE (wzdłuż osi Y)
            self.target_angle = 90.0
            self.target_x = 400.0
            self.target_y = 150.0
            self.obs1_x = self.target_x + 80.0
            self.obs1_y = self.target_y
            self.obs2_x = self.target_x - 80.0
            self.obs2_y = self.target_y

        self.step_count = 0

        # Zapamiętujemy startowy dystans do obliczania nagrody za postęp
        self.prev_dist = np.sqrt(
            (self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2
        )

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1

        # --- 1. FIZYKA I PORUSZANIE SIĘ ---
        if action == 0:
            self.car_speed += 0.5
        elif action == 1:
            self.car_speed -= 0.5

        self.car_speed = max(-5.0, min(5.0, self.car_speed))

        # Skręcanie działa tylko wtedy, gdy auto jedzie
        if abs(self.car_speed) > 0.1:
            if action == 2:
                self.car_angle += 3.0
            elif action == 3:
                self.car_angle -= 3.0

        # Trygonometria - przeliczanie kąta i prędkości na pozycje X i Y
        rad = np.deg2rad(self.car_angle)
        self.car_x += self.car_speed * np.cos(rad)
        self.car_y -= self.car_speed * np.sin(rad)  # oś Y rośnie w dół w PyGame

        # --- 2. OBLICZANIE METRYK ---
        dist = np.sqrt((self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2)

        angle_diff = abs((self.car_angle % 360) - self.target_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        dist_to_obs1 = np.sqrt((self.car_x - self.obs1_x)**2 + (self.car_y - self.obs1_y)**2)
        dist_to_obs2 = np.sqrt((self.car_x - self.obs2_x)**2 + (self.car_y - self.obs2_y)**2)

        # --- 3. REWARD SHAPING ---
        # Bazowa kara za każdy krok (motywuje do szybkiego działania)
        reward = -0.05

        # NAGRODA ZA POSTĘP: jeśli agent zbliżył się do celu, dostaje nagrodę
        dist_improvement = self.prev_dist - dist
        reward += dist_improvement * 0.3  # im większy postęp, tym więcej
        self.prev_dist = dist

        # NAGRODA ZA DOBRY KĄT: gdy agent jest blisko, liczy się też orientacja
        if dist < 150:
            reward -= angle_diff * 0.004  # kara rośnie z bliskością celu

        # KARA ZA ZBLIŻANIE SIĘ DO PRZESZKÓD (strefa ostrzeżenia: 80px)
        if dist_to_obs1 < 80:
            reward -= (80 - dist_to_obs1) * 0.01
        if dist_to_obs2 < 80:
            reward -= (80 - dist_to_obs2) * 0.01

        terminated = False
        truncated = False

        # WARUNEK ZWYCIĘSTWA
        if dist < 5 and angle_diff < 3 and abs(self.car_speed) < 0.5:
            reward = 200.0
            terminated = True

        # WARUNEK PORAŻKI: wyjazd poza ekran
        if self.car_x < 0 or self.car_x > 800 or self.car_y < 0 or self.car_y > 600:
            reward = -100.0
            terminated = True

        # WARUNEK PORAŻKI: kolizja z przeszkodą
        if dist_to_obs1 < 50 or dist_to_obs2 < 50:
            reward = -100.0
            terminated = True

        # Limit kroków
        if self.step_count >= 600:
            truncated = True

        observation = self._get_observation()
        info = {"dist": dist, "angle_diff": angle_diff}

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Autonomiczne Parkowanie - Środowisko RL")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Tło - asfalt
        self.window.fill((50, 50, 50))

        # Rysuj linie parkingowe (biały wzór)
        for i in range(0, 800, 40):
            pygame.draw.line(self.window, (80, 80, 80), (i, 0), (i, 600), 1)

        # Miejsce parkingowe (META) - zielona ramka
        if self.target_angle == 0.0:
            spot_width, spot_height = 120, 60
        else:
            spot_width, spot_height = 60, 120

        target_rect = pygame.Rect(
            self.target_x - spot_width // 2,
            self.target_y - spot_height // 2,
            spot_width,
            spot_height
        )
        pygame.draw.rect(self.window, (0, 200, 0), target_rect, 3)

        # Litera "P" w środku miejsca parkingowego
        font = pygame.font.SysFont(None, 36)
        p_text = font.render("P", True, (0, 200, 0))
        self.window.blit(p_text, (self.target_x - 8, self.target_y - 12))

        # Przeszkody (czerwone samochody)
        if self.target_angle == 0.0:
            obs_w, obs_h = 100, 40
        else:
            obs_w, obs_h = 40, 100

        obs1_rect = pygame.Rect(self.obs1_x - obs_w // 2, self.obs1_y - obs_h // 2, obs_w, obs_h)
        obs2_rect = pygame.Rect(self.obs2_x - obs_w // 2, self.obs2_y - obs_h // 2, obs_w, obs_h)
        pygame.draw.rect(self.window, (200, 50, 50), obs1_rect)
        pygame.draw.rect(self.window, (200, 50, 50), obs2_rect)

        # Samochód agenta (niebieski, obracany)
        car_width, car_height = 80, 40
        car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
        car_surface.fill((50, 150, 255))
        pygame.draw.rect(car_surface, (255, 0, 0), (car_width - 15, 0, 15, car_height))

        rotated_car = pygame.transform.rotate(car_surface, self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.window.blit(rotated_car, car_rect.topleft)

        # HUD - informacje na ekranie
        font_small = pygame.font.SysFont(None, 24)
        dist = np.sqrt((self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2)
        angle_diff = abs((self.car_angle % 360) - self.target_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        hud_texts = [
            f"Dystans do celu: {dist:.1f} px",
            f"Roznica kata: {angle_diff:.1f} deg",
            f"Predkosc: {self.car_speed:.2f}",
            f"Krok: {self.step_count}",
        ]
        for i, text in enumerate(hud_texts):
            surf = font_small.render(text, True, (255, 255, 255))
            self.window.blit(surf, (10, 10 + i * 22))

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None