# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for x in range(self.iterations):
            newValue = util.Counter()
            for state in states:
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    qValues = []
                    for action in actions:
                        qValues.append(self.getQValue(state,action))
                    newValue[state] = max(qValues)
            
            for state in newValue:
                self.values[state] = newValue[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0.0

        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            q += prob * (reward + self.discount * self.values[nextState])
        return q  

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)

        if not actions:
            return None
        best_action, best_value = None, float('-inf')
        
        for a in actions:
            q = self.computeQValueFromValues(state, a)
            if q > best_value:
                best_value, best_action = q, a
        
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        if not states:
            return
        n = len(states)
        
        for k in range(self.iterations):
            s = states[k % n]
            
            if self.mdp.isTerminal(s):
                continue
            actions = self.mdp.getPossibleActions(s)
            
            if not actions:
                self.values[s] = 0.0
                continue
            
            best = float('-inf')
            
            for a in actions:
                q = 0.0
                
                for ns, p in self.mdp.getTransitionStatesAndProbs(s, a):
                    r = self.mdp.getReward(s, a, ns)
                    q += p * (r + self.discount * self.values[ns])
                
                if q > best:
                    best = q
            
            self.values[s] = best

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        """Prioritized Sweeping Value Iteration."""
        states = self.mdp.getStates()

        # 1) Compute predecessors: for each state, the set of states that can reach it
        predecessors = {s: set() for s in states}
        for s in states:
            for a in self.mdp.getPossibleActions(s):
                for ns, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                    if prob > 0:
                        predecessors.setdefault(ns, set()).add(s)

        # Helper to compute max_a Q(s,a)
        def best_q(s):
            actions = self.mdp.getPossibleActions(s)
            if not actions:
                return 0.0
            m = float('-inf')
            for a in actions:
                q = 0.0
                for ns, p in self.mdp.getTransitionStatesAndProbs(s, a):
                    r = self.mdp.getReward(s, a, ns)
                    q += p * (r + self.discount * self.values[ns])
                if q > m:
                    m = q
            return m

        # 2) Initialize priority queue with -diff priorities (states in given order)
        pq = util.PriorityQueue()
        for s in states:
            if self.mdp.isTerminal(s):
                continue
            diff = abs(self.values[s] - best_q(s))
            pq.update(s, -diff)

        # 3) Iteratively pop highest-priority state, update, then push predecessors
        for _ in range(self.iterations):
            if pq.isEmpty():
                break

            s = pq.pop()

            if not self.mdp.isTerminal(s):
                self.values[s] = best_q(s)

            for p in predecessors.get(s, ()):
                if self.mdp.isTerminal(p):
                    continue
                diff_p = abs(self.values[p] - best_q(p))
                if diff_p > self.theta:
                    pq.update(p, -diff_p)
        

