import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from parking_env import ParkingEnv

class SimpleLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, checkpoint_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.checkpoint_freq = checkpoint_freq
        self.log_file = "postepy_ai.txt"
        self.best_mean_reward = -np.inf

        os.makedirs("checkpoints", exist_ok=True)

        with open(self.log_file, "w") as f:
            f.write("--- LOG TRENINGU AUTONOMICZNEGO PARKOWANIA ---\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                successes = sum(1 for ep in self.model.ep_info_buffer if ep["r"] > 150)
                success_rate = successes / len(self.model.ep_info_buffer) * 100

                log_text = (
                    f"Krok: {self.num_timesteps:07d} | "
                    f"Srednia nagroda: {mean_reward:.2f} | "
                    f"Skutecznosc: {success_rate:.1f}%"
                )
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    log_text += " *** NOWY REKORD ***"

                print(log_text)
                with open(self.log_file, "a") as f:
                    f.write(log_text + "\n")

        if self.n_calls % self.checkpoint_freq == 0:
            step_label = f"{self.num_timesteps:07d}"
            model_path = f"checkpoints/model_{step_label}"
            norm_path = f"checkpoints/vec_normalize_{step_label}.pkl"

            self.model.save(model_path)
            self.training_env.save(norm_path)
            self.training_env.save("vec_normalize.pkl")

            print(f"  >> Checkpoint: {model_path}.zip")

        return True


def make_env():
    """Fabryka środowisk - potrzebna do VecEnv."""
    def _init():
        env = ParkingEnv(render_mode=None)
        env = Monitor(env)
        return env
    return _init

N_ENVS = 4
env = DummyVecEnv([make_env() for _ in range(N_ENVS)])

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

eval_env = DummyVecEnv([make_env()])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)


model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    # Sieć: dwie warstwy po 256 neuronów - wystarczy do tego problemu
    policy_kwargs=dict(net_arch=[256, 256]),
    # Rozmiar próbki zbieranej przed aktualizacją sieci
    n_steps=2048,
    # Rozmiar mini-batcha do optymalizacji
    batch_size=256,
    # Liczba epok optymalizacji na zebranej próbce
    n_epochs=10,
    # Współczynnik uczenia
    learning_rate=3e-4,
    # Gamma - jak bardzo AI ceni przyszłe nagrody (wysoka = dalekowzroczna)
    gamma=0.99,
    # Clip epsilon - jak bardzo agresywnie zmieniamy politykę
    clip_range=0.2,
    # Entropia - zachęca do eksploracji (nie utknij w lokalnym optimum)
    ent_coef=0.005,
    # Normalizacja błędu przewidywania wartości
    vf_coef=0.5,
    # Gradient clipping - zapobiega eksplozji gradientów
    max_grad_norm=0.5,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=max(10_000 // N_ENVS, 1),
    n_eval_episodes=20,
    deterministic=True,
    verbose=0,
)

logger_callback = SimpleLoggerCallback(check_freq=5_000, checkpoint_freq=20_000)

TOTAL_STEPS = 1_500_000

print("=" * 60)
print("  TRENING AUTONOMICZNEGO PARKOWANIA")
print("=" * 60)
print(f"  Srodowiska rownolegle: {N_ENVS}")
print(f"  Laczna liczba krokow: {TOTAL_STEPS:,}")
print(f"  Architektura sieci: [256, 256]")
print(f"  Postep zapisywany do: postepy_ai.txt")
print(f"  Najlepszy model: ./best_model/best_model.zip")
print("=" * 60)

try:
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[logger_callback, eval_callback],
        progress_bar=True,
    )
    print("\nTrening zakończony pomyślnie!")
except KeyboardInterrupt:
    print("\n\nPrzerwano przez użytkownika (Ctrl+C). Zapisuję model...")

model.save("parking_ai_model")
env.save("vec_normalize.pkl")

print("Zapisano: parking_ai_model.zip")
print("Zapisano: vec_normalize.pkl")

env.close()
eval_env.close()