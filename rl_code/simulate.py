from Agents.agents import Random_Agent


def simulate_game(game_obj, agent_types=('Random', 'Random')):
    print("---Top of simulate game---")
    print("Game object: %s" % str(game_obj))

    agents = []
    for i, typ in enumerate(agent_types):
        if typ == 'Random':
            agents.append(Random_Agent(id=i))
    print(agents)


    print("End simulation")