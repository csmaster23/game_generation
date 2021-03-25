from Agents.agents import Random_Agent


def simulate_game(game_obj, agent_types=('Random', 'Random')):
    print("---Top of simulate game---")
    print("Game object: %s" % str(game_obj))

    agents = []
    for i, typ in enumerate(agent_types):
        if typ == 'Random':
            agents.append(Random_Agent(id=i+1))

    game_finished = False
    while game_finished is False:
        curr_state = game_obj.get_game_state()                                      # get current game state
        print("Player %s Turn" % str(curr_state['turn']))
        curr_agent = get_agent_by_name(curr_state['turn'], agents)

        curr_state, curr_agent, game_obj = simulate_turn(curr_state, curr_agent, game_obj)

        game_finished, players_still_in_game = game_obj.check_game_over()           # check if game is over
        print("Game Finished Bool: %s" % str(game_finished))
        curr_state['turn'] = update_turn(curr_state, game_obj)                      # this changes turn to next player
        game_obj.set_game_state(curr_state)                                         # set the game_obj to correct state


    print("End simulation")


def simulate_turn(state, agent, game):
    print("--top of simulating a turn for agent: %s" % str(agent.id))
    action_dict = game.get_actions_for_player(state['turn'])
    agent_choice = agent.choose_action(action_dict)

    # TODO: Carry out action that has been chosen by the agent

    return state, agent, game

def update_turn(curr_state, game_obj):
    parsed = curr_state['turn'].split("_")
    num = int(parsed[-1])
    num += 1
    if num > game_obj.num_players:
        num = 1
    ss = "player_" + str(num)
    return ss

def get_agent_by_name(name, agents):
    for agent in agents:
        if agent.player_name == name:
            return agent