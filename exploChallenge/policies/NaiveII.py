__author__ = 'ftruzzi'

# Naive2 solution based on "Linear Bayes Policy for Learning in Contextual-Bandits"
# but not using the pursuit method of calculation

import random
import numpy as np
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
import random as pr


def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

if __name__ == '__main__':
    test = [0,1,2,2]
    for i in range(10):
        print rargmax(test)


class Naive2(ContextualBanditPolicy):
    def __init__(self):
        self.clicks = {}
        self.selections = {}

    def getActionToPerform(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.clicks:
                self.clicks[article.getID()] = 1.0
                self.selections[article.getID()] = 1.0

        clicks = [self.clicks[article.getID()] for article in possibleActions]
        selections = [self.selections[article.getID()] for article in possibleActions]

        #action = possibleActions[np.argmax([(1.0 * x) / y for x, y in zip(clicks, selections)])]
        #k = [(1.0 * x) / y for x, y in zip(clicks, selections)]


        action = possibleActions[np.argmax([np.random.beta(20*self.clicks[article.getID()],
                                                           20*(self.selections[article.getID()] - self.clicks[
                                                               article.getID()] + 1)) for article in possibleActions])]

        return action

    def updatePolicy(self, c, a, reward):
        self.selections[a.getID()] += 1
        if reward is True:
            self.clicks[a.getID()] += 1
