
from abc import ABC, abstractmethod
import time
import random
from collections import defaultdict
from node import Edge, Node


class Policy(ABC):
    """Abstract class for agents"""

    def __init__(self):
        pass
    
    def choose_action(self, state):
        """Abstract function for choosing a action given the state of the game."""
        pass
    


class RandomPolicy(Policy):
    """Random agent, chooses a random legal action"""
    def __init__(self):
      self.name = "Random Policy"

    def choose_action(self, state):
        _,_,_, moves = state.possible_actions()
        chosen_move = random.choice(moves)
        return chosen_move


class GreedyPolicy(Policy):
    """Greedy agent, chooses a the lowest ranking trick in the trick category with the highest amount of cards."""
    def __init__(self):
      self.name = "Greedy Agent"
    def choose_action(self, state):
        singles, pairs, fives, all_moves = state.possible_actions()
        if len(fives) > 0:
          return fives[0]
        elif len(pairs) > 0:
          return pairs[0]
        elif len(singles) > 0:
          return singles[0]
        return all_moves[0]
        
#Note: This MCTS policy was created by me last semester for CPSC474's assignent 4, which was also about MCTS.
#I only made minimal changes to ensure it fit with the Big2 environment's functions and to add time_limit decay
class MCTSPolicy(Policy):
  def __init__(self, time_limit, player_num, time_decay_rate):
    """Monte Carlo Tree Search agent, creates and iterates through the tree, then chooses best action based on payoff per visit
    time_limit: double - the amount of time in seconds the agent has to implement the tree search
    time_decay_rate: double -  decay rate of the time_limit. Multiplied every round"""
    self.name = "MCTS Agent"
    self.time_limit = time_limit
    self.player_num = player_num
    self.time_decay_rate = time_decay_rate

  def choose_action(self, state):
    _,_,_,actions = state.possible_actions()
    num_actions = len(actions)
    # print("Possible actions: ", num_actions)
    # print("Cards left: ", state.cards_left())
    if num_actions == 1:
      return actions[0]

    adjusted_time = self.time_limit * 0.96

    adjusted_time = adjusted_time * (self.time_decay_rate ** (state.num_turn - 1))

    start_time = time.time()
    root = Node(state)
    tot_traversals = 0
    while time.time() - start_time < adjusted_time:
        tot_traversals +=1
        self.traverse(root)
        visits = [edge.visits for edge in root.edges]
        if max(visits) > 300:
          break
    chosen_action = root.get_best_action()
    if chosen_action is None:
      chosen_action = actions[0]
    return chosen_action

  def traverse(self, starting_state):
      root = starting_state
      s = root
      path = []

      #go from root to leaf node and add all of them to path list
      while not s.is_leaf():
          s, edge = s.get_best_child()
          path.append(edge)

      #leaf to random child that was picked should be last edge in path
      if not s.state.is_terminal():
          s.make_edges()
          s, edge = s.get_random_child()
          path.append(edge)

      #go from leaf state to terminal state using random choices
      s = s.get_random_terminal()
      terminal_payoff = s.state.is_winner(self.player_num)
      #update the payoff and num_visits for every edge in the path
      for edge in path:
          edge.update_stats(terminal_payoff)




class RulePolicy(Policy):
    """Rule based agent, chooses a card based on the game state and hueristics.
    Note: this agent was not designed by me. It is heavily based off of Sugiyanto et al.'s agent, with some changes."""
    def __init__(self):
      self.name = "Rule Based Agent"

    def choose_action(self, state):
        oppo_singles, oppo_pairs, oppo_fives, oppo_moves = state.opponent_actions()
        num_cards = state.num_cards_agent()
        oppo_num_cards = state.num_cards_opponents()
        singles, pairs, fives, all_moves =  state.possible_actions()
        in_control = False
        if state.last_trick is None:
          in_control = True
        #can only pass, so pass
        if len(all_moves) == 1:
            return all_moves[0]

        class_a, class_b, class_c, class_d = state.return_classes(singles, pairs, fives, oppo_singles, oppo_pairs, oppo_fives)
        if in_control:

          #if agent has 2 cards
          if num_cards == 2:
            if len(pairs) == 1:
              return pairs[0]
            #print(all_moves)
            elif len(class_a) > 0 or 1 in oppo_num_cards:
              return singles[-1]
            else:
              return singles[0]

          if num_cards == 3:
            single_ranks = get_ranks_present(singles)
            pair_ranks = get_ranks_present(pairs)
            single_rank = [single_rank for single_rank in single_ranks if single_ranks not in pair_ranks]
            #if a pair exists
            if len(pairs) > 0:
              for pair in pairs:
                if pair in class_a:
                  return pair
              for single in singles:
                if single in class_a:
                  return single
              if 1 in oppo_num_cards:
                return pairs[0]
              elif 2 in oppo_num_cards:
                return singles[0]
              else:
                if len(class_d) > 0:
                  return class_d[0]
                elif len(class_c) > 0:
                  return class_c[0]
                elif len(class_b) > 0:
                  return class_b[0]
                elif len(class_a) > 0:
                  return class_a[0]
            #if 3 singles
            else:
              if singles[2] in class_a:
                return singles[1]
              elif 1 in oppo_num_cards:
                return singles[-1]
              else:
                return singles[0]

          if num_cards == 4:
            single_ranks = get_ranks_present(singles)
            pair_ranks = get_ranks_present(pairs)
            #if there are pairs
            if len(pairs) > 0:
              #two pairs
              if len(pairs) == 2:
                if pairs[-1] in class_a or 2 in oppo_num_cards:
                  return pairs[-1]
                else:
                  return pairs[0]
              #one pair and two singles
              if len(pairs) == 1:
                for trick in singles:
                  if trick.cards[0].rank == pair_ranks[0]:
                    singles.remove(trick)
                if len(class_a) > 0:
                  if singles[-1] in class_a:
                    return singles[0]
                  elif 1 in oppo_num_cards:
                    return pairs[-1]
                  elif 2 in oppo_num_cards:
                    return singles[0]
                  else:
                    return singles[0]
                else:
                  if len(class_d) > 0:
                    return class_d[0]
                  elif len(class_c) > 0:
                    return class_c[0]
                  elif len(class_b) > 0:
                    return class_b[0]
                  elif len(class_a) > 0:
                    return class_a[0]
            #4 singles
            else:
              if len(class_a) > 0:
                return singles[1]
              elif 1 in oppo_num_cards:
                return singles[-1]
              else :
                return singles[0]

          #more than 4 cards:
          if num_cards > 4:
            if 1 in oppo_num_cards:
              if len(fives) > 0:
                return fives[0]
              elif len(pairs) > 0:
                return pairs[0]
              else:
                return singles[-1]
            elif len(fives) == 0 and len(pairs) > len(singles):
              return pairs[0]
            else:
              if len(class_d) > 0:
                return class_d[0]
              elif len(class_c) > 0:
                return class_c[0]
              elif len(class_b) > 0:
                return class_b[0]
              elif len(class_a) > 0:
                return class_a[0]
        #not in control
        else:
          hand_size = len(state.player_hands[state.player_control])
          for move in class_d:
            if not hold_back(move, num_cards, oppo_num_cards, class_a, class_b, class_c, class_d, fives, len(state.discard_pile), state.num_passes):
              return move
          for move in class_c:
            if not hold_back(move, num_cards, oppo_num_cards, class_a, class_b, class_c, class_d, fives, len(state.discard_pile), state.num_passes):
              return move
          for move in class_b:
            if not hold_back(move, num_cards, oppo_num_cards, class_a, class_b, class_c, class_d, fives, len(state.discard_pile), state.num_passes):
              return move
          for move in class_a:
            if not hold_back(move, num_cards, oppo_num_cards, class_a, class_b, class_c, class_d, fives, len(state.discard_pile), state.num_passes):
              return move
          if min(oppo_num_cards) == 1:
            return all_moves[0]
          # else:
          #   return all_moves[-1]


        if min(oppo_num_cards) == 1:
          return all_moves[0]
        elif len(fives) > 0:
          return fives[0]
        elif len(pairs) > 0:
          return pairs[0]
        elif len(singles) > 0:
          return singles[0]
        return all_moves[0]

def get_ranks_present(moves):
  ranks = set()
  for move in moves:
    new_ranks = set(move.get_ranks())
    ranks = ranks | new_ranks
  return list(ranks)

def hold_back(move, hand_size, oppo_num_cards, class_a, class_b, class_c, class_d, fives, num_cards_history, num_passes):
  """helper function for rule based agent. Used to determine if an trick should be held back due to important cards that would be used"""
  #considering single
  move_size = len(move.cards)
  if move_size == 1:
    if hand_size <= 2:
      return False
    elif 2 in oppo_num_cards or 1 in oppo_num_cards:
      return False
    elif (len(class_a) < len(class_b) + len(class_c) + len(class_d) or min(oppo_num_cards) > 6) and move in class_a :
      return True
    else:
      return False
  # considering a pair
  if move_size == 2:
    if hand_size <= 3:
      return False
    elif min(oppo_num_cards) > 2 and move.cards[0].rank == 13:
      return True
    else:
      return False

  #considering fives
  if move_size == 5:
    if hand_size == 5:
      return False
    elif min(oppo_num_cards) > 6 and num_cards_history < 15 and num_passes >  0:
      #and len(fives) == 2 and (move in class_a) or (move in class_b)
      return True
    else:
      return False








