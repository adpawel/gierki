import pygame
from parking_env import ParkingEnv

pygame.init()

env = ParkingEnv(render_mode="human")
obs, info = env.reset()

env.render()

running = True
print("--- STEROWANIE ---")
print("Strzałka w GÓRĘ: Gaz")
print("Strzałka w DÓŁ: Hamulec/Cofanie")
print("Strzałki LEWO/PRAWO: Skręcanie")
print("Zamknij okno, aby wyjść.")

while running:
    action = 4 
    
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_UP]:
        action = 0
    elif keys[pygame.K_DOWN]:
        action = 1
    elif keys[pygame.K_LEFT]:
        action = 2
    elif keys[pygame.K_RIGHT]:
        action = 3

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()

    if terminated or truncated:
        if reward > 50:
            print(f"SUKCES! Zaparkowałeś! Nagroda: {reward}")
        else:
            print(f"PORAŻKA! Nagroda: {reward}")
            
        obs, info = env.reset()
        pygame.time.wait(1000)

env.close()