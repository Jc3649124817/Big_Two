
from deck import Deck
import numpy as np
import torch
import time
import random
from DQN import DQN
from policy import GreedyPolicy, RandomPolicy, MCTSPolicy, RulePolicy
from Big2 import Big2

def states_to_vecs(states, player_num):
  if type(states) is not list:
    states = [states]
  feature_vec = np.zeros((len(states), 60))
  #print(states)
  for idx, state in enumerate(states):
    deck = Deck()
    for card in deck.cards:
      #encodes hand, discard pile, and opponents' remaining cards
      idx2 = card.rank - 1
      if card in state.player_hands[player_num]:
        feature_vec[idx][idx2] +=0.25
      elif card in state.discard_pile:
        feature_vec[idx][idx2 + 13 * 1] +=0.25
      else:
        feature_vec[idx][idx2 + 13 * 2] +=0.25
    trick = state.last_trick
    #encodes last trick played
    if trick is None:
      feature_vec[idx][13*3 + 16] = 1
    else:
      if trick.trick_type == "Single":
        feature_vec[idx][13*3 ] = 1
      elif trick.trick_type == "Pair":
        feature_vec[idx][13*3 + 1] = 1
      else:
        feature_vec[idx][13*3 + 2] = 1
      for card in trick.cards:
        feature_vec[idx][13*3 + 2 + card.rank] += 0.25
    #encodes hand sizes of agent and opponents
    player_num_card = player_num
    for j in range(4):
      feature_vec[idx][13*3 + 17 + j] = len(state.player_hands[(player_num_card + j) % 4]) / 13
  return torch.from_numpy(feature_vec).float()


#0 to 12 is single, 13 to 25 is pairs, 26 to 27 are fives, 28 is pass logits_order should be 29 length
def logits_to_action(state, logits_order, random=False):
  """Given a tensor of logits, Find the first legal move represented by a logit
  logits_order: torch.tensor[int]
  random: boolean - shuffle logit to choose a random legal action instead, being of epsilon."""
  logits_order = logits_order.cpu().numpy()
  if random:
    np.random.shuffle(logits_order)

  singles, pairs, fives, all_moves = state.possible_actions()
  for i in range(logits_order.shape[1]):

  #for logit in logits_order:
    logit = logits_order[0, i]
    #singles
    if logit < 13 and len(singles) > 0:
      desired_rank = logits_order[0, i] + 1
      for trick in singles:
        if trick.cards[0].rank == desired_rank:
          return trick, logit
    #pairs
    elif logit>=13 and logit < 26 and len(pairs) > 0:
      desired_rank = logit + 1 % 13
      for trick in pairs:
        if trick.cards[0].rank == desired_rank:
          return trick, logit
    #fives
    elif logit == 26 and len(fives) > 0:
      return fives[0], logit
    elif logit == 27 and len(fives) > 0:
      return fives[-1], logit
    #pass
    elif logit == 28:
      return all_moves[-1], logit




def train(model,  num_episodes, optimizer, device, epsilon, epsilon_decay_rate, epsilon_min, oppo_agent, decay_rate, decay_rate_period, checkpoint_path, custom_dataset=None):
    """Training function for the DQN model
    num_episodes: int - if a preexisting dataset doesn't exist, simulate this number of games using Monte Carlo Simulations
    optimizer: - chosen optimizer for gradient descent
    epsilon: chance of the DQN choosing a random action
    epsilon_decay_rate: how much epsilon decays throughout time
    epsilon_min: the minimum epsilon allowed before decay stops
    oppo_agent: opponent chosen to be used during Monte Carlo Simualtions
    decay_rate: alpha decay rate
    decay_rate_period: how often the alpha and epsilon decay rates are used. Measured in number of episodes
    checkpoint_path: Where to store the model
    custom_dataset: custom dataset to train model on. Model runs until all beginning states are used at least once"""
    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)
    # TODO: Move model to device using .to()
    model = model.to(device)
    # target_model.load_state_dict(model.state_dict())
    gamma = 0.95
    loss_func = torch.nn.SmoothL1Loss()


    time_start = time.time()

    #run episodes
    for episode in range(min(num_episodes, len(custom_dataset))):
        #decay learning rate when needed
        if episode % decay_rate_period == 0 and episode != 0:
            for param_group in optimizer.param_groups:
              param_group["lr"] *= decay_rate
            epsilon = max(epsilon_min, epsilon*epsilon_decay_rate)


        states = []
        actions_done = []
        rewards = []
        game_over = False
        num_turns = random.random()
        if custom_dataset is None:
          game_state = Big2().deal_all_cards()
        else:
            game_state = custom_dataset[episode]
        game_state = game_state.sim_turns( random.randint(0,3), agent=oppo_agent, epsilon=epsilon)
        if game_state.is_terminal():
          break

        player_num = game_state.player_control
        #Run the episode and store states, and chosen actions
        while not game_over:
          states.append(game_state)

          action = None
          state_features =  states_to_vecs(game_state, player_num).to(device)

          _,_,_, actions = game_state.possible_actions()

          logits = model(state_features)
          chosen_logit = None
          sorted_logits = torch.argsort(logits, dim=1, descending=True)

          if random.random() < epsilon:
            action, chosen_logit = logits_to_action(game_state, sorted_logits, random=True)
          else:
            action, chosen_logit = logits_to_action(game_state, sorted_logits)
          #print("chosen_logit: ", chosen_logit)
          actions_done.append(chosen_logit)
          game_state = game_state.successorMCTS(action)
          game_over = game_state.is_terminal()

        #episode has ended, get terminal reward and backpropogate it using discount rate gamma
        reward = game_state.terminal_reward(player_num)
        rewards = np.zeros(len(actions_done))
        for i in range(len(actions_done) - 1, -1, -1):
          rewards[i] = reward
          reward = reward * gamma


        #calculates the q-value predicted by the model for the given state, actions pairs in the episode
        state_features = states_to_vecs(states, player_num).to(device)
        actions_done = torch.tensor(actions_done).to(device).unsqueeze(1)
        q_values = model(state_features)
        chosen_q_values = model(state_features).gather(1, actions_done).squeeze(1)
        rewards = torch.tensor(rewards).to(device)

        #calculate loss and backpropogate
        optimizer.zero_grad()
        loss = loss_func(chosen_q_values, rewards)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        
        #save model once in a while in case runtime crashes
        if episode % 1000 == 0:
          time_elapsed = time.time() - time_start
          time_start = time.time()
          print(f"Episode {episode} loss: {loss} Time Elapsed: {time_elapsed}")
          torch.save({
              "episodes" : num_episodes,
              "model_state_dict" : model.state_dict(),
              "optimizer" : optimizer.state_dict()
            }, checkpoint_path)

    #save model at end of training
    torch.save({
        "episodes" : num_episodes,
        "model_state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }, checkpoint_path)

def make_game_dataset(n_games):
  games_dataset = []
  for i in range(n_games):
    games_dataset.append(Big2().deal_all_cards())
  return games_dataset

class DQNPolicy():
  def __init__(self, checkpoint, device = "cuda"):
    self.device = device
    self.name = "DQN Agent"
    self.model = DQN()
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.model.to(device)

  def choose_action(self, state):
    features = states_to_vecs(state, state.player_control).to(self.device)
    logits = self.model(features)
    sorted_logits = torch.argsort(logits, dim=1, descending=True)
    _,_,_,possible_actions = state.possible_actions()
    action, logit = logits_to_action(state, sorted_logits)
    return action

if __name__ == "__main__":
    """Set hyperparameters for the DQN and train it and evaluate."""
    learning_rate = 0.001
    learning_decay_rate = 0.99
    learning_decay_rate_period = 10000
    
    #custom_dataset  = make_game_dataset(1000000)

    model = DQN()

    device = "cuda"
    epsilon = 0.9
    epsilon_decay_rate = 0.95
    epsilon_min = 0.3
    oppo_agent = GreedyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    checkpoint_path = './checkpoint-{}.pth'.format("DQN")

    # model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    train(model, 100000000, optimizer, device, epsilon, epsilon_decay_rate, 
          epsilon_min, oppo_agent, learning_decay_rate, learning_decay_rate_period, checkpoint_path, custom_dataset=None)
    

    #compares 
