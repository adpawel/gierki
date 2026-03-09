from easyAI import AI_Player, Negamax
from utils import run_tournament
from models import NegamaxNoAB

if __name__ == "__main__":
    ai_ab_4 = AI_Player(Negamax(4))
    ai_no_ab_4 = AI_Player(NegamaxNoAB(4))
    
    ai_ab_5 = AI_Player(Negamax(5))
    ai_no_ab_5 = AI_Player(NegamaxNoAB(5))

    games = 20

    print("================ DETERMINISTYCZNY NIM ================\n")
    run_tournament(ai_ab_4, ai_no_ab_4, "Negamax(AB, depth=4)", "Negamax(NoAB, depth=4)", False, games)
    run_tournament(ai_ab_5, ai_no_ab_5, "Negamax(AB, depth=5)", "Negamax(NoAB, depth=5)", False, games)
    
    print("================ PROBABILISTYCZNY NIMBY (10%) ================\n")
    run_tournament(ai_ab_4, ai_no_ab_4, "Negamax(AB, depth=4)", "Negamax(NoAB, depth=4)", True, games)
    run_tournament(ai_ab_5, ai_no_ab_5, "Negamax(AB, depth=5)", "Negamax(NoAB, depth=5)", True, games)

    print("================ RÓŻNE GŁĘBOKOŚCI (Alfa-Beta) ================\n")
    run_tournament(ai_ab_4, ai_ab_5, "Negamax(AB, depth=4)", "Negamax(AB, depth=5)", True, 20)