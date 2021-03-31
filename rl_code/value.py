from rl_code.simulate import simulate_game
import numpy as np

def get_value_basic(game_obj, combined_turns=100, num_simulations=100):
    all_results = []

    for _ in range(num_simulations):
        results_dict = simulate_game(game_obj, agent_types=('Random', 'Random'), combined_turns=combined_turns, verbose=False)

        if not results_dict["game_finished"]:
            all_results.append(results_dict["turns"] / combined_turns)
        else:
            all_results.append(1.0)

    return np.mean(all_results) * 2 - 1