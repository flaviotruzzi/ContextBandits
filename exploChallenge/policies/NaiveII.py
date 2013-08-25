__author__ = 'ftruzzi'

# Naive Bayes based on "Linear Bayes Policy for Learning in Contextual-Bandits"
# but not using the pursuit method of calculation

import random
import numpy as np
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy


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

        action = possibleActions[np.argmax([(1.0 * x) / y for x, y in zip(clicks, selections)])]

        return action

    def updatePolicy(self, c, a, reward):
        self.selections[a.getID()] += 1
        if reward is True:
            self.clicks[a.getID()] += 1
