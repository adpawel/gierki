import json
from easyAI import AI_Player, Negamax
from utils import add_to_report
from models import NegamaxNoAB

if __name__ == "__main__":
    ai_ab_4 = AI_Player(Negamax(4))
    ai_no_ab_4 = AI_Player(NegamaxNoAB(4))
    
    ai_ab_5 = AI_Player(Negamax(5))
    ai_no_ab_5 = AI_Player(NegamaxNoAB(5))

    games = 20
    json_report = []

    print("================ DETERMINISTYCZNY NIM ================\n")
    add_to_report(json_report, "Deterministyczny Nim", ai_ab_4, ai_no_ab_4, "Negamax(AB, depth=4)", "Negamax(NoAB, depth=4)", False, games)
    add_to_report(json_report, "Deterministyczny Nim", ai_ab_5, ai_no_ab_5, "Negamax(AB, depth=5)", "Negamax(NoAB, depth=5)", False, games)
    
    print("================ PROBABILISTYCZNY NIMBY (10%) ================\n")
    add_to_report(json_report, "Probabilistyczny Nimby", ai_ab_4, ai_no_ab_4, "Negamax(AB, depth=4)", "Negamax(NoAB, depth=4)", True, games)
    add_to_report(json_report, "Probabilistyczny Nimby", ai_ab_5, ai_no_ab_5, "Negamax(AB, depth=5)", "Negamax(NoAB, depth=5)", True, games)

    print("================ RÓŻNE GŁĘBOKOŚCI (Alfa-Beta) ================\n")
    add_to_report(json_report, "Probabilistyczny Nimby", ai_ab_4, ai_ab_5, "Negamax(AB, depth=4)", "Negamax(AB, depth=5)", True, 20)

    with open("negamaxAB_results.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=4, ensure_ascii=False)
        