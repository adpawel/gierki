import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
import pygame
import pygame.gfxdraw # Używane do gładszych kół (krzewów)
import numpy as np


class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32
        )

        self.prev_dist = None

    def _get_observation(self):
        """Buduje znormalizowany wektor obserwacji."""
        dist = np.sqrt((self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2)
        norm_dist = np.clip(dist / 1000.0, 0.0, 1.0)
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

        self.car_x = 400.0 + random.uniform(-30, 30)
        self.car_y = 500.0 + random.uniform(-20, 20)
        self.car_angle = random.uniform(-10, 10)
        self.car_speed = 0.0

        parking_type = random.choice([0, 1])

        if parking_type == 0:
            self.target_angle = 0.0
            self.target_x = 400.0
            self.target_y = 100.0
            self.obs1_x = self.target_x + 140.0
            self.obs1_y = self.target_y
            self.obs2_x = self.target_x - 140.0
            self.obs2_y = self.target_y
        else:
            self.target_angle = 90.0
            self.target_x = 400.0
            self.target_y = 150.0
            self.obs1_x = self.target_x + 80.0
            self.obs1_y = self.target_y
            self.obs2_x = self.target_x - 80.0
            self.obs2_y = self.target_y

        self.step_count = 0

        self.prev_dist = np.sqrt(
            (self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2
        )

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1

        if action == 0:
            self.car_speed += 0.5
        elif action == 1:
            self.car_speed -= 0.5

        self.car_speed = max(-5.0, min(5.0, self.car_speed))

        if abs(self.car_speed) > 0.1:
            if action == 2:
                self.car_angle += 3.0
            elif action == 3:
                self.car_angle -= 3.0

        rad = np.deg2rad(self.car_angle)
        self.car_x += self.car_speed * np.cos(rad)
        self.car_y -= self.car_speed * np.sin(rad)  # oś Y rośnie w dół w PyGame

        dist = np.sqrt((self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2)

        angle_diff = abs((self.car_angle % 360) - self.target_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        dist_to_obs1 = np.sqrt((self.car_x - self.obs1_x)**2 + (self.car_y - self.obs1_y)**2)
        dist_to_obs2 = np.sqrt((self.car_x - self.obs2_x)**2 + (self.car_y - self.obs2_y)**2)

        reward = -0.05

        dist_improvement = self.prev_dist - dist
        reward += dist_improvement * 0.3  # im większy postęp, tym więcej
        self.prev_dist = dist

        if dist < 150:
            reward -= angle_diff * 0.004  # kara rośnie z bliskością celu

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

        def draw_detailed_car(surface, color, w, h, is_agent=False):
            """Rysuje samochód z detalami (zawsze w poziomie na powierzchni)."""
            # Korpus
            pygame.draw.rect(surface, color, (0, 0, w, h), border_radius=7)
            # Szyby
            pygame.draw.rect(surface, (40, 45, 50), (w * 0.6, 5, w * 0.2, h - 10), border_radius=2)
            pygame.draw.rect(surface, (40, 45, 50), (w * 0.2, 5, w * 0.1, h - 10), border_radius=1)
            # Światła przednie
            pygame.draw.rect(surface, (255, 255, 200), (w - 6, 2, 6, 10), border_radius=2)
            pygame.draw.rect(surface, (255, 255, 200), (w - 6, h - 12, 6, 10), border_radius=2)
            # Światła tylne
            pygame.draw.rect(surface, (180, 0, 0), (0, 2, 4, 10), border_radius=1)
            pygame.draw.rect(surface, (180, 0, 0), (0, h - 12, 4, 10), border_radius=1)
            
            if is_agent: 
                pygame.draw.rect(surface, (255, 255, 255, 100), (w//2 - 5, 0, 10, h), 1)

        self.window.fill((45, 90, 45))
        
        pygame.draw.rect(self.window, (35, 35, 35), (100, 0, 600, 600))
        
        for i in range(100, 700, 50):
            pygame.draw.line(self.window, (45, 45, 45), (i, 0), (i, 600), 1)

        side_colors = [(120, 120, 120), (50, 50, 50), (30, 60, 140), (160, 120, 30), (180, 180, 180)]
        decor_w, decor_h = 85, 42
        
        for i in range(7):
            y_pos = 25 + i * 85
            l_surf = pygame.Surface((decor_w, decor_h), pygame.SRCALPHA)
            draw_detailed_car(l_surf, side_colors[i % len(side_colors)], decor_w, decor_h)
            self.window.blit(l_surf, (10, y_pos))
            
            r_surf = pygame.Surface((decor_w, decor_h), pygame.SRCALPHA)
            draw_detailed_car(r_surf, side_colors[(i + 2) % len(side_colors)], decor_w, decor_h)
            r_surf = pygame.transform.rotate(r_surf, 180)
            self.window.blit(r_surf, (705, y_pos))

        if self.target_angle == 0.0:
            spot_w, spot_h = 130, 70
        else:
            spot_w, spot_h = 70, 130

        target_rect = pygame.Rect(self.target_x - spot_w//2, self.target_y - spot_h//2, spot_w, spot_h)
        pygame.draw.rect(self.window, (40, 65, 40), target_rect.inflate(-4, -4), border_radius=5)
        pygame.draw.rect(self.window, (0, 255, 100), target_rect, 2, border_radius=5)

        font_p = pygame.font.SysFont("Arial", 36, bold=True)
        p_text = font_p.render("P", True, (0, 255, 100))
        self.window.blit(p_text, (self.target_x - 12, self.target_y - 20))

        obs_base_w, obs_base_h = 100, 45
        for ox, oy in [(self.obs1_x, self.obs1_y), (self.obs2_x, self.obs2_y)]:
            obs_surf = pygame.Surface((obs_base_w, obs_base_h), pygame.SRCALPHA)
            draw_detailed_car(obs_surf, (180, 50, 50), obs_base_w, obs_base_h)
            
            rotated_obs = pygame.transform.rotate(obs_surf, self.target_angle)
            obs_rect = rotated_obs.get_rect(center=(ox, oy))
            self.window.blit(rotated_obs, obs_rect.topleft)

        car_base_w, car_base_h = 80, 40
        car_surf = pygame.Surface((car_base_w, car_base_h), pygame.SRCALPHA)
        draw_detailed_car(car_surf, (50, 130, 255), car_base_w, car_base_h, is_agent=True)

        rotated_car = pygame.transform.rotate(car_surf, self.car_angle)
        car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.window.blit(rotated_car, car_rect.topleft)

        # --- 6. HUD ---
        hud_bg = pygame.Surface((230, 110), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 170))
        self.window.blit(hud_bg, (110, 10)) # Przesunięty na początek asfaltu

        font_small = pygame.font.SysFont("Consolas", 18)
        dist = np.sqrt((self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2)
        angle_diff = abs((self.car_angle % 360) - self.target_angle)
        if angle_diff > 180: angle_diff = 360 - angle_diff

        hud_texts = [
            f"Dystans: {dist:6.1f} px",
            f"Kat:     {angle_diff:6.1f} deg",
            f"Predkosc:{self.car_speed:6.2f}",
            f"Krok:    {self.step_count:4d}",
        ]
        for i, text in enumerate(hud_texts):
            surf = font_small.render(text, True, (220, 220, 220))
            self.window.blit(surf, (120, 20 + i * 22))

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])