from easyAI import AI_Player, Negamax
from utils import run_tournament
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results_det, results_prob, matchups, games_total):
    """Generates a bar chart comparing results for different depths."""
    labels = [f"Depth {m[0]} vs {m[1]}" for m in matchups]
    
    det_p1_wins = [res[1] for res in results_det]
    det_p2_wins = [res[2] for res in results_det]
    
    prob_p1_wins = [res[1] for res in results_prob]
    prob_p2_wins = [res[2] for res in results_prob]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Figure 1 - Deterministic
    rects1 = ax1.bar(x - width/2, det_p1_wins, width, label='Słabszy (Mniejsza głębokość)', color='salmon')
    rects2 = ax1.bar(x + width/2, det_p2_wins, width, label='Silniejszy (Większa głębokość)', color='skyblue')
    ax1.set_ylabel('Liczba wygranych')
    ax1.set_title(f'Wariant Deterministyczny (Nim)\n{games_total} gier na starcie')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # Figure 2 - Probabilistic
    rects3 = ax2.bar(x - width/2, prob_p1_wins, width, label='Słabszy (Mniejsza głębokość)', color='salmon')
    rects4 = ax2.bar(x + width/2, prob_p2_wins, width, label='Silniejszy (Większa głębokość)', color='skyblue')
    ax2.set_ylabel('Liczba wygranych')
    ax2.set_title(f'Wariant Probabilistyczny (Nimby 10%)\n{games_total} gier na starcie')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()

    for ax in [ax1, ax2]:
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('wykres1.png', dpi=300)
    print("\nZapisano wykres jako 'wykres_depth.png'")
    plt.show()


if __name__ == "__main__":
    games_total = 100
    matchups = [(3, 3), (3, 4), (3, 5), (4, 5)]
    
    results_det = []
    results_prob = []
    
    print(f"Rozpoczynam serię turniejów (po {games_total} gier w starciu)...")
    
    for d1, d2 in matchups:
        ai1 = AI_Player(Negamax(d1))
        ai2 = AI_Player(Negamax(d2))
        
        print(f"\n--- Starcie: Depth {d1} vs Depth {d2} ---")
        
        print("Wersja deterministyczna:")
        res_d = run_tournament(ai1, ai2, f"Depth {d1}", f"Depth {d2}", is_probabilistic=False, num_games=games_total)
        results_det.append(res_d)
        
        print("Wersja probabilistyczna (Nimby):")
        res_p = run_tournament(ai1, ai2, f"Depth {d1}", f"Depth {d2}", is_probabilistic=True, num_games=games_total)
        results_prob.append(res_p)
        
    plot_results(results_det, results_prob, matchups, games_total)