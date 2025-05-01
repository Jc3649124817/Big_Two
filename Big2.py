import itertools as it
import random
import numpy as np
from collections import defaultdict
import copy
from deck import Deck
from trick import Trick

class Big2:
    """Creates a game state object for Big Two
    playerhands: a list for each player with up to 13 Card Objects
    las_trick: the last trick played by the previous player. Is None if it is a new round
    player_control: The player number of the player whose turn it is"""
    def __init__(self):
        self.player_hands = [[] for _ in range(4)]
        self.discard_pile = []
        self.last_trick = None
        self.player_control = random.randint(0,3)
        self.num_passes = 0
        self.num_turn = 1

    def __deepcopy__(self, memo):
      """Creates a deep copy of Big2(), mainly for Tree Searches were the original states need to be maintained"""
      new_state = Big2()
      new_state.player_hands = [[card for card in player_hand] for player_hand in self.player_hands]
      new_state.last_trick = self.last_trick
      new_state.player_control = self.player_control
      new_state.num_passes = self.num_passes
      new_state.discard_pile = self.discard_pile
      new_state.num_turn = self.num_turn
      return new_state

    def cards_left(self):
      """returns the total number of cards in players hands."""
      cards_left = 0
      for i in range(4):
        cards_left+= len(self.player_hands[i])
      return cards_left

    def print_game(self):
        print(f"Big Two Game - player turn: {self.player_control}")
        for i in range(4):
            string = ""
            for j in range(len(self.player_hands[i])):
                string = string + str(self.player_hands[i][j].rank) + ", "
            print(f"Player {i}: {string}")


    def deal_all_cards(self):
        """Shuffles the deck and deals all the cards to the players. 13 cards per player."""
        deck = Deck()
        deck.shuffle()
        for i in range(13):
            for j in range(4):
                self.player_hands[j].append(deck.deal())
        return self
        # for i in range(4):
          #print(self.player_hands[i])

    def successorMCTS(self, action):
        """The successor function that takes an action by player x then returns the state when its player x's turn again, assuming the other agents are random
        action: Trick object"""
        new_state = copy.deepcopy(self)
        #print("In SuccMCTS action: ", action)
        if action.trick_type != "Pass":
            new_state.player_hands[self.player_control] =  [card for card in new_state.player_hands[new_state.player_control] if card not in action.cards]
            new_state.last_trick = action
            for card in action.cards:
                new_state.discard_pile.append(card)
            new_state.num_passes = 0
        else:
            new_state.num_passes +=1
            if new_state.num_passes == 3:
                new_state.last_trick = None
                new_state.num_passes = 0
        new_state.player_control = (new_state.player_control + 1) % 4
        if new_state.player_control == 0:
          new_state.num_turn+= 1
        #iterates up to 3 more times to get the next state MCTS needs, uses random agents for the other 3.
        other_turns = 0
        while not new_state.is_terminal() and other_turns < 3:
            other_turns+=1
            _,_,_, actions = new_state.possible_actions(new_state)
            action = random.choice(actions)
            new_state = new_state.successor(action)
        return new_state



    def successor(self, action):
        """The successor function that takes an action by player x then returns the immediate state that results from the action. 
        action: Trick object"""
        new_state = copy.deepcopy(self)
        if action.trick_type != "Pass":
            new_state.player_hands[self.player_control] =  [card for card in new_state.player_hands[new_state.player_control] if card not in action.cards]
            new_state.last_trick = action
            for card in action.cards:
                new_state.discard_pile.append(card)
            new_state.num_passes = 0
        else:
            new_state.num_passes +=1
            if new_state.num_passes == 3:
                new_state.last_trick = None
                new_state.num_passes = 0
        new_state.player_control = (new_state.player_control + 1) % 4
        if new_state.player_control == 0:
          new_state.num_turn+= 1
        #new_state.print_game()
        return new_state

    def sim_whole_game(self, action, agent=None, epsilon = 0):
      """The successor function that takes an action by player x then returns the end of the game.
        action: Trick object
        agent: Agent object -  The agent used for the rest of the simulated game
        epsilon: int - The chance of random choice being used instead of inputted agent"""
      new_state = copy.deepcopy(self)
      while not new_state.is_terminal():
          _,_,_, actions = new_state.possible_actions(new_state)
          if agent is None or random.random() < epsilon:
            action = random.choice(actions)
          else:
            action = agent.choose_action(new_state)
          new_state = new_state.successor(action)
      return new_state


    
    def sim_turns(self, n_turns, agent=None, epsilon = 0):
      """The successor function that takes an action by player x then returns the end of the game.
        action: Trick object
        agent: Agent object -  The agent used for the rest of the simulated game
        epsilon: int - The chance of random choice being used instead of inputted agent"""
      new_state = copy.deepcopy(self)
      turns_elapsed = 0
      while not new_state.is_terminal() and turns_elapsed < n_turns:
          _,_,_, actions = new_state.possible_actions(new_state)
          if agent is None or random.random() < epsilon:
            action = random.choice(actions)
          else:
            action = agent.choose_action(new_state)
          new_state = new_state.successor(action)
          turns_elapsed+=1
      return new_state



    def is_terminal(self):
        """Determines if the game is over"""
        for i in range(4):
            if len(self.player_hands[i]) == 0:
                return True
        return False

    def calculate_rewards(self, new_state, player_num):
      """Given a successor state, returns the reward from transition from the original state to the successor state"""
      #0.1 for every card discarded
      cards_discarded = len(self.player_hands[player_num]) - len(new_state.player_hands[player_num])
      #0.2 if you won round with your last discard
      if new_state.last_trick is None:
        won_round = 0.1
      else:
        won_round = 0
      terminal_reward = 0
      #if game over, reward for winning and additional rewards/punishment for every card remaining 
      if new_state.is_terminal() and new_state.winner() == player_num:
        terminal_reward += 1 +  new_state.terminal_points(player_num) * 0.05
      elif new_state.is_terminal() and new_state.winner() != player_num:
        terminal_reward +=  0 - new_state.terminal_points(player_num) * 0.05
      
      return 0.05 * cards_discarded + won_round + terminal_reward


    def winner(self):
        """Given a terminal state, returns the winner"""
        if not self.is_terminal():
            return -1
        for i in range(4):
            if len(self.player_hands[i]) == 0:
                return i
        print("Is winning function for Big Two failed")

    def terminal_reward(self, player_num):
      """Returns the reward for winning or losing"""
      if len(self.player_hands[player_num]) == 0:
        return 1
      else:
        return -1

    def is_winner(self, player_num):
      """Given a terminal state and player number, determines if the player won or not."""
      if len(self.player_hands[player_num]) == 0:
        return 1
      else:
        return 0

    def terminal_points(self, player_num):
      """Given a terminal state, determines how many points you gained or loss with the traditional system"""
      num_opponent_cards = 0
      if len(self.player_hands[player_num]) == 0:
        player_array = [0,1,2,3]
        player_array.remove(player_num)
        for i in player_array:
          num_opponent_cards+= len(self.player_hands[i])
        return num_opponent_cards
      else:
        return -1 * len(self.player_hands[player_num])




    def possible_actions(self, opponent_cards = [], opponent=False, constrained = True):
        """For the state, determines all the possible actions the agent can take. Can also calculate potential opponent moves.
        opponent_cards: list(Cards) - used to calculate opponents
        opponent: Boolean - if True, this function calculates opponent's possible moves instead of agents.
        constrained: boolean - if True, some combinations of cards are illegal so they need to be filtered out"""

        hand = self.player_hands[self.player_control]

        if opponent:
          hand = opponent_cards

        ranks_dict = defaultdict(list)
        for card in hand:
            ranks_dict[card.rank].append(card)

        singles = [[card] for card in hand ]
        pairs = [pair[:2] for pair in ranks_dict.values() if len(pair) >= 2]
        triples = [pair[:3] for pair in ranks_dict.values() if len(pair) >= 3]
        quads = [pair[:4] for pair in ranks_dict.values() if len(pair) >= 4]

        full_houses = []
        bombs = []
        straights = []

        #make all full houses
        for triple in triples:
            for pair in pairs:
                pair_rank = pair[0].rank
                triple_rank = pair[0].rank
                if pair_rank != triple_rank:
                    full_house = triple + pair
                    full_house = Trick(full_house, "Full House")
                    full_houses.append(full_house)
        #makes all bombs
        for quad in quads:
            for single in singles:
                quad_rank = quad[0].rank
                single_rank = single[0].rank
                if  quad_rank != single_rank:
                    bomb = quad  + single
                    bomb  = Trick(bomb, "Bomb")
                    bombs.append(bomb)

        unique_ranks = sorted(set([card.rank for card in hand]))
        #make all possible straights
        for i in range(len(unique_ranks) - 4):
            straight_range = unique_ranks[i: i+5]
            #print(straight_range)
            if straight_range[4] - straight_range[0] == 4:
                straight = []
                for rank in straight_range:
                    straight.append(ranks_dict[rank][0])
                straight = Trick(straight, "Straight")
                straights.append(straight)
        #do this now that that pairs aren't needed for full house
        singles = sorted([Trick(single, "Single") for single in singles])
        pairs = sorted([Trick(pair, "Pair") for pair in pairs])
        fives = sorted(full_houses + bombs + straights)

        #returns 1, 2, 5 card tricks, plus all of them together

        #all moves are possible b/c you are in control
        if self.last_trick is None or not constrained:
          all_actions = singles + pairs + fives
          return singles, pairs, fives, all_actions
        #someone already played cards so you are constrained
        else:
          singles = self.filter_actions(singles)
          pairs = self.filter_actions(pairs)
          fives = self.filter_actions(fives)
          all_actions = singles + pairs + fives + [Trick([], "Pass")]
          return singles, pairs, fives, all_actions


    def filter_actions(self, all_actions):
        """Given a list of actions, determines which ones are legal and returns them
        all_actions: list(Trick)"""
        constrained_actions = [action for action in all_actions if (action.num_cards == self.last_trick.num_cards and self.last_trick < action)]
        return constrained_actions

    def return_classes(self, singles, pairs, fives, oppo_singles, oppo_pairs, oppo_fives):
      """Given a list of agents actions and possible opponent actions, 
      Categorizes agent's actions into 4 classes (a strongest, d weakest)
      Used for rule based agent"""
      class_a, class_b, class_c, class_d = [], [] , [], []
      #class b must be stronger than 70% of moves
      class_b_percent = 0.7

      all_moves = (sorted(fives), sorted(pairs), sorted(singles))
      all_oppo_moves = (oppo_fives, oppo_pairs, oppo_singles)

      for moves, oppo_moves in zip(all_moves, all_oppo_moves):
        num_oppo_moves = len(oppo_moves) + 1
        for move in moves:
          oppo_moves.append(move)
          oppo_moves = sorted(oppo_moves)
          move_index = oppo_moves.index(move)
          if move.always_beatable() or move_index == 0:
            class_d.append(move)
          elif move.is_unbeatable() or move_index >= len(oppo_moves) - 1:
            class_a.append(move)
          elif move_index / num_oppo_moves > class_b_percent:
            class_b.append(move)
          else:
            class_c.append(move)
          oppo_moves.remove(move)
      return class_a, class_b, class_c, class_d

    def opponent_actions(self):
      """Given a state, returns all possible actions by opponents"""
      player_nums = [0, 1, 2, 3]
      player_nums.remove(self.player_control)
      opponent_cards = []
      for player_num in player_nums:
        for card in self.player_hands[player_num]:
          opponent_cards.append(card)
      return self.possible_actions(opponent_cards, True, False)


    def num_cards_agent(self):
      return len(self.player_hands[self.player_control])

    def num_cards_opponents(self):
      num_cards = [len(self.player_hands[x])  for x in [0,1,2,3]]
      num_cards.pop(self.player_control)
      return num_cards

def play(policies, num_games):
    """Simulates games to determine which agent is better
    policies: list(Agent) - The 4 agents that are being tested
    num_games: int - the number of games simulated"""
    games_won = [0, 0, 0, 0]
    points_total = [0,0,0,0]
    turns = 0

    for i in range(num_games):
        game_over = False
        game_state = Big2()
        game_state.deal_all_cards()

        while not game_over:
            turns+=1
            action = policies[game_state.player_control].choose_action(game_state)
            game_state = game_state.successor(action)
            game_over = game_state.is_terminal()
        games_won[game_state.winner()] += 1
        for player_num in range(4):
          points_total[player_num] += game_state.terminal_points(player_num)

        if i % 100 == 0 and i > 0:
          print(f"Total games played: {i}")
          print(games_won)
          for i in range(4):
            print(f"Model {i} - {policies[i].name}, Games won: {games_won[i]}, Points won: {points_total[i]}")

    turns_per_game = turns / num_games
    print("Avg turns: ",  turns_per_game)
    print(f"Total games played: {num_games}")
    print(games_won)
    for i in range(4):
        print(f"Model {i} - {policies[i].name}, Games won: {games_won[i]}, Points won: {points_total[i]}")

def play_dataset(policies, games_dataset):
    """Simulates games to determine which agent is better with a specific dataset
    policies: list(Agent) - The 4 agents that are being tested
    dataset: list(Big2) - dataset used"""
    games_won = [0, 0, 0, 0]
    points_total = [0,0,0,0]
    turns = 0
    n_games = len(games_dataset)
    for i in range(n_games):
        game_over = False
        game_state = games_dataset[i]

        while not game_over:
            turns+=1
            #game_state.print_game()
            action = policies[game_state.player_control].choose_action(game_state)
            #print("chosen action: ", action)
            game_state = game_state.successor(action)
            game_over = game_state.is_terminal()
        games_won[game_state.winner()] += 1
        for player_num in range(4):
          points_total[player_num] += game_state.terminal_points(player_num)

        if i % 100 == 0 and i > 0:
          print(f"Total games played: {i}")
          print(games_won)
          for i in range(4):
            print(f"Model {i} - {policies[i].name}, Games won: {games_won[i]}, Points won: {points_total[i]}")

    turns_per_game = turns / n_games
    print("Avg turns: ",  turns_per_game)
    print(f"Total games played: {n_games}")
    print(games_won)
    for i in range(4):
        print(f"Model {i} - {policies[i].name}, Games won: {games_won[i]}, Points won: {points_total[i]}")