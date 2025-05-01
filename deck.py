import itertools as it
import random


class Card:
    """Creates a card with the given suit and rank
    rank = rank of the card converted to 1-13 (lowest to highest rank)
    suit = suit of the card in str (ex. "C" for club)"""
    def __init__(self, rank, suit=0):
        self.rank = rank
        self.suit = suit


    def suit(self):
        return self.suit

    def get_rank(self):
        return self.rank


    def __repr__(self):
        string = "[" + str(self.rank) + str(self.suit) + "]"
        return string

    def __eq__(self, other):
        if self.rank == other.rank  and self.suit == other.suit:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.rank > other.rank:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.rank < other.rank:
            return True
        else:
            return False

    def __le__(self, other):
        if self.rank == other.rank or self.rank < other.rank:
            return True
        else:
            return False

class Deck:
    """Creates a deck with 1 copy of all the 52 possible cards"""
    def __init__(self, ranks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], suits = ["D", "H", "C", "S"]):
        self.cards = []
        for combo in it.product(ranks, suits):
            self.cards.append(Card(combo[0], combo[1]))
        # for card in self.cards:
        #   print(card.rank, card.suit)

    """Shuffles the deck, used to create different shuffles for game states of Big Two"""
    def shuffle(self):
        random.shuffle(self.cards)

    """Removes one card from the Deck's top"""
    def deal(self):
        card = self.cards.pop()
        return card