from easyAI import AI_Player, Negamax
from utils import run_tournament

if __name__ == "__main__":
    ai_depth_3 = AI_Player(Negamax(3))
    # Upewnij się, że drugi gracz to depth=5 (miałeś 3 w oryginalnym kodzie)
    ai_depth_5 = AI_Player(Negamax(5)) 
    
    games_total = 100
    
    print(f"Rozpoczynam turniej ({games_total} gier w każdym wariancie)...\n")
    
    print("--- TEST 1: Deterministyczny Nim (Zwykły) ---")
    run_tournament(ai_depth_3, ai_depth_5, "Negamax(depth=3)", "Negamax(depth=5)", is_probabilistic=False, num_games=games_total)
    
    print("--- TEST 2: Probabilistyczny Nimby (10% szansy na błąd) ---")
    run_tournament(ai_depth_3, ai_depth_5, "Negamax(depth=3)", "Negamax(depth=5)", is_probabilistic=True, num_games=games_total)