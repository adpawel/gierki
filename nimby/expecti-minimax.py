import json
from easyAI import AI_Player, Negamax
from utils import add_to_report
from models import ExpectiNegamax

if __name__ == "__main__":
    ai_negamax_4 = AI_Player(Negamax(4))
    ai_expecti_4 = AI_Player(ExpectiNegamax(4))
    
    ai_negamax_5 = AI_Player(Negamax(5))
    ai_expecti_5 = AI_Player(ExpectiNegamax(5))

    games = 100
    json_report = []

    print("================ NEGAMAX vs EXPECTIMINIMAX ================\n")
    print("Testowane na probabilistycznej grze (Nimby 10%)...")
    
    add_to_report(json_report, "Probabilistyczny Nimby (10%)", ai_negamax_4, ai_expecti_4, 
                  "Zwykły Negamax(4)", "ExpectiMinimax(4)", True, games)
                  
    add_to_report(json_report, "Probabilistyczny Nimby (10%)", ai_negamax_5, ai_expecti_5, 
                  "Zwykły Negamax(5)", "ExpectiMinimax(5)", True, games)

    with open("expectiminimax-results.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=4, ensure_ascii=False)
        