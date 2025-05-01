from Big2 import Big2, play_game_state
from deck import Deck
import numpy as np
from policy import GreedyPolicy
import torch
from torch.utils.data import Dataset, DataLoader

class DataSetObject():
  """
  Creates a DataSetObject by running a simulations of games of Big 2. 
  The starting states are encoded into a feature vector for x, and the winners at the terminal states recorded for y
  int(n_samples): the number of datapoints you want in the dataset.

  """
  def __init__(self, n_samples):
    self.policies = [GreedyPolicy(), GreedyPolicy(), GreedyPolicy(), GreedyPolicy()]
    self.x = []
    self.y = []
    self.classes = [0,1,2,3]
    self.create_samples(n_samples)

  def create_samples(self, n_samples):
    game_states = []
    for i in range(n_samples):
      # if i % 10 == 9:
      #   print(i)
      game_state = Big2().deal_all_cards()
      game_states.append(game_state)
    self.x = self.states_to_vecs(game_states)
    self.y = self.get_winners(game_states)
      
  def states_to_vecs(self, states):
    #time_start = time.time()
    if type(states) is not list:
      states = [states]
    feature_vec = np.zeros((len(states), 56))
    deck = Deck()
    #for every state, make a feature vectors
    for idx, state in enumerate(states):
      for player_num in range(0,4):
        for card in deck.cards:
          #encodes hand, discard pile, and opponents' remaining cards
          idx2 = card.rank - 1
          if card in state.player_hands[player_num]:
            feature_vec[idx][idx2 + player_num * 13] +=0.25
        if state.player_control == player_num:
            feature_vec[idx][player_num + 52] = 1
    #print(time.time() - time_start)
    return torch.from_numpy(feature_vec).float()

  def get_winners(self, game_states):
    winners = []
    for state in game_states:
      games_won = play_game_state(self.policies, state, 1, False)
      winners.append(games_won.index(1))
    return torch.tensor(winners).long()
  

class Big2Dataset(Dataset):
    """This is an child class of the torch Dataset class. You use the x and y from your DataSetObject class
      as the inputs to x and y to make the Big2Dataset class, which can be used with the DataLoader from torch.util.data for mini-batching"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        datapoint = (self.x[idx], self.y[idx])
        return datapoint