import os
import sys
import glob
import pygame
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from parking_env import ParkingEnv

# -------------------------------------------------------
# Wybór modelu do załadowania
# Użycie:
#   python play.py              -> ładuje najlepszy / finalny model
#   python play.py best         -> ładuje best_model/best_model.zip
#   python play.py checkpoint   -> pokazuje listę checkpointów do wyboru
#   python play.py 0040000      -> ładuje checkpoints/model_0040000.zip
# -------------------------------------------------------

def select_model():
    arg = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if arg == "checkpoint":
        pliki = sorted(glob.glob("checkpoints/model_*.zip"))
        if not pliki:
            print("Brak checkpointów w folderze checkpoints/")
            sys.exit(1)
        print("\nDostępne checkpointy:")
        for i, p in enumerate(pliki):
            print(f"  [{i}] {p}")
        wybor = int(input("Wybierz numer: "))
        model_path = pliki[wybor].replace(".zip", "")
        step = os.path.basename(model_path).replace("model_", "")
        norm_path = f"checkpoints/vec_normalize_{step}.pkl"
        return model_path, norm_path

    elif arg == "best":
        return "best_model/best_model", "vec_normalize.pkl"

    elif arg == "auto":
        if os.path.exists("best_model/best_model.zip"):
            return "best_model/best_model", "vec_normalize.pkl"
        return "parking_ai_model", "vec_normalize.pkl"

    else:
        model_path = f"checkpoints/model_{arg}"
        norm_path = f"checkpoints/vec_normalize_{arg}.pkl"
        return model_path, norm_path


print("Ładowanie modelu...")
model_path, norm_path = select_model()

if not os.path.exists(model_path + ".zip"):
    print(f"Nie znaleziono modelu: {model_path}.zip")
    sys.exit(1)

model = PPO.load(model_path)
print(f"Załadowano model: {model_path}.zip")

def make_env():
    def _init():
        env = ParkingEnv(render_mode="human")
        return Monitor(env)
    return _init

env = DummyVecEnv([make_env()])

if os.path.exists(norm_path):
    env = VecNormalize.load(norm_path, env)
    env.training = False
    env.norm_reward = False
    print(f"Załadowano statystyki normalizacji: {norm_path}")
else:
    print("UWAGA: Nie znaleziono vec_normalize.pkl - model może działać gorzej!")

N_EPISODES = 10
wins = 0

print(f"\nTestowanie agenta przez {N_EPISODES} epizodów...")
print("-" * 40)

for epizod in range(N_EPISODES):
    obs = env.reset()
    env.env_method("render")

    suma_nagrod = 0.0
    kroki = 0

    done = [False]
    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        suma_nagrod += reward[0]
        kroki += 1

        env.env_method("render")

        time.sleep(0.03)

    wynik = "SUKCES" if suma_nagrod > 150 else "PORAŻKA"
    if suma_nagrod > 150:
        wins += 1

    print(f"Epizod {epizod + 1:2d}: {wynik} | Nagroda: {suma_nagrod:7.2f} | Kroków: {kroki}")
    time.sleep(1.0)

print("-" * 40)
print(f"Skuteczność: {wins}/{N_EPISODES} ({wins/N_EPISODES*100:.0f}%)")

env.close()