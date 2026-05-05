import pygame
from parking_env import ParkingEnv

# 1. Dodajemy inicjalizację PyGame na samym początku!
pygame.init()

# Tworzymy środowisko
env = ParkingEnv(render_mode="human")
obs, info = env.reset()

# 2. Wywołujemy render RAZ przed pętlą, aby okno się poprawnie zbudowało
env.render()

running = True
print("--- STEROWANIE ---")
print("Strzałka w GÓRĘ: Gaz")
print("Strzałka w DÓŁ: Hamulec/Cofanie")
print("Strzałki LEWO/PRAWO: Skręcanie")
print("Zamknij okno, aby wyjść.")

# Główna pętla gry
while running:
    action = 4 
    
    # Teraz event.pump() i get_pressed() zadziałają bez błędu
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

    # Obsługa zamykania okna krzyżykiem
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Wykonanie kroku w środowisku
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Rysowanie klatki
    env.render()

    # Sprawdzenie warunków końca gry (Sukces lub wyjazd za mapę)
    if terminated or truncated:
        if reward > 50:
            print(f"SUKCES! Zaparkowałeś! Nagroda: {reward}")
        else:
            print(f"PORAŻKA! Nagroda: {reward}")
            
        # Resetujemy grę, żeby spróbować jeszcze raz
        obs, info = env.reset()
        pygame.time.wait(1000)

env.close()