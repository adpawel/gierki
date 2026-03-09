from easyAI import AI_Player, Negamax
from utils import run_tournament
from models import ExpectiNegamax

if __name__ == "__main__":
    ai_negamax_4 = AI_Player(Negamax(4))
    ai_expecti_4 = AI_Player(ExpectiNegamax(4))
    
    ai_negamax_5 = AI_Player(Negamax(5))
    ai_expecti_5 = AI_Player(ExpectiNegamax(5))

    games = 100

    print("================ STARCIĘ TYTANÓW: NEGAMAX vs EXPECTIMINIMAX ================\n")
    print("Testowane na PRAWDZIWEJ, probabilistycznej grze (Nimby 10%)...")
    
    run_tournament(ai_negamax_4, ai_expecti_4, "Zwykły Negamax(4)", "ExpectiMinimax(4)", is_probabilistic=True, num_games=games)
    run_tournament(ai_negamax_5, ai_expecti_5, "Zwykły Negamax(5)", "ExpectiMinimax(5)", is_probabilistic=True, num_games=games)