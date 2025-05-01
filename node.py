import math
import random
import time


#Note: This class was created by me last semester for CPSC474's assignent 4, which was also about MCTS.
#I only made minimal changes to ensure it fit with the Big2 environment's functions

#edge that represents the shift from one move to another, and the visits/payoff stats of a state
class Edge():
    def __init__(self, action, child_node) -> None:
        self.action = action
        self.child_node = child_node
        self.payoff = 0
        self.visits = 0

    #adds payoff and visits  to the edge's stats
    def update_stats(self, payoff):
        self.payoff += payoff
        self.visits += 1

#node class that represents a state and possible actions through edges.
class Node():
    def __init__(self, state) -> None:
        self.state = state
        self.edges = []

    #is the state currently presented as a leaf in current MCTS tree?
    def is_leaf(self):
        return (len(self.edges) == 0)

    #creates edges based on possible actions in this state
    def make_edges(self):
        _,_,_,actions = self.state.possible_actions()
        #print("Making EDges: ", actions)
        for action in actions:
            self.edges.append(Edge(action, Node(self.state.successorMCTS(action))))


    #choose a random edge and give the successor state
    def get_random_child(self):
        edge = random.choice(self.edges)
        child =  edge.child_node
        return (child, edge)


    #get the best child edge using the UCB formula depending on if the actor is player 0 or 1
    def get_best_child(self):
        total_visits = 0

        edges = self.edges
        edge_scores = [None] * len(self.edges)
        #print(len(self.edges))
        #print(len(edge_scores))
        best_index = None
        for edge in edges:
            total_visits += edge.visits

        for edge, index in zip(edges, range(len(self.edges))):
            if edge.visits == 0:
                best_index = index
                return self.edges[best_index].child_node, self.edges[best_index]
            edge_scores[index] = edge.payoff/edge.visits + math.sqrt(2*math.log(total_visits)  / edge.visits )

        best_index = edge_scores.index(max(edge_scores))
        return self.edges[best_index].child_node, self.edges[best_index]



    #returns the best action for the root using average payoff, depends on if actor is player 0 or 1
    def get_best_action(self):
        best_payoff = None
        best_action = None

        for edge in self.edges:
            if edge.visits == 0:
              break
            else:
              avg_payoff = edge.payoff/edge.visits
            if best_payoff == None or avg_payoff > best_payoff:
                best_action = edge.action
                #print(best_action)
                best_payoff = avg_payoff
        #print(best_action, "best action")
        return best_action


    #get terminal state from a Node object with making any Node objects or edges along the way
    def get_random_terminal(self):
        state = self.state
        while not state.is_terminal():
            _,_,_,actions = state.possible_actions()
            random_action = random.choice(actions)
            state = state.successor(random_action)
        node = Node(state)
        return node



