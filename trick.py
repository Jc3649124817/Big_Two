
import random
import itertools as it
from deck import Card

class Trick:
    """" a  trick given a list of Card objects, doubles as the action object for the environment.
    cards: list(Card) - the cards that make up the trick
    trick_type: str - the type of trick this is"""
    def __init__(self, cards, trick_type):
        self.cards = cards
        self.num_cards = len(cards)
        #ordering from weakest to strongest
        self.trick_ordering = ["Straight", "Full House", "Bomb"]
        self.trick_type = trick_type


    def __lt__(self, other):
        my_cards = self.cards
        their_cards = other.cards
        #pass is weakest
        if self.num_cards == 0:
          return True
        elif other.num_cards == 0:
          return False

        #compare singles or pairs using rank of first card
        elif self.num_cards == 1 or self.num_cards == 2:
          return my_cards[0].rank < their_cards[0].rank

        #compare 5 card trick type strength
        my_type = self.trick_ordering.index(self.trick_type)
        their_type = self.trick_ordering.index(other.trick_type)
        if my_type < my_type:
            return True
        elif my_type > my_type:
            return False

        #same trick type so need to evaluate strength of indiviual cards
        else:
            for i in range(5):
              if my_cards[i] < their_cards[i]:
                return True
        return False

    def get_ranks(self):
      ranks = [card.rank for card in self.cards]
      return ranks

    def is_unbeatable(self):
      """determines if its possible for a trick to beat this trick"""
      start_rank = self.cards[0].rank
      if self.trick_type == "Single" and start_rank == 13:
        return True
      elif self.trick_type == "Double" and start_rank == 13:
        return True
      elif self.trick_type == "Bomb" and start_rank == 13:
        return True
      else:
        return False

    def always_beatable(self):
      """determines if its possible for this trick to beat any other trick"""
      start_rank = self.cards[0].rank
      if self.trick_type == "Single" and start_rank == 1:
        return True
      elif self.trick_type == "Double" and start_rank == 1:
        return True
      elif self.trick_type == "Straight" and start_rank == 1:
        return True
      else:
        return False


    def __repr__(self):
      string = "Trick: "
      for card in self.cards:
        string = string +  repr(card) + " "
      string = string +  "-" + self.trick_type
      return string

