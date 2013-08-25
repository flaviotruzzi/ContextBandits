__author__ = 'ftruzzi'

# Naive Bayes III based on "Linear Bayes Policy for Learning in Contextual-Bandits"
# but not using the pursuit method of calculation

import numpy as np
from collections import defaultdict
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy


class Naive3(ContextualBanditPolicy):
    def __init__(self):
        self.clicks = {}
        self.selections = {}

    def getActionToPerform(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.clicks:
                self.clicks[article.getID()] = defaultdict(int)
                self.selections[article.getID()] = defaultdict(int)

        sumOfclicks = [sum(self.clicks[article.getID()].values() + [1]) for article in possibleActions]
        sumOfSelections = [sum(self.selections[article.getID()].values() + [1]) for article in possibleActions]

        action = possibleActions[np.argmax([(1.0 * x) / y for x, y in zip(sumOfclicks, sumOfSelections)])]

        return action

    def updatePolicy(self, c, a, reward):
        for f, p in enumerate(c.getFeatures()):
            if p is 1:
                if reward is True:
                    self.clicks[a.getID()][f] += 1.0
                self.selections[a.getID()][f] += 1.0
