__author__ = 'ftruzzi'

# Naive Bayes III based on "Linear Bayes Policy for Learning in Contextual-Bandits"
# but not using the persuit method of calculation

import numpy as np
from collections import defaultdict
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy


class NaiveBayes(ContextualBanditPolicy):

    def __init__(self):
        self.clicks = {}
        self.selections = {}

    def getActionToPerform(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.clicks:
                self.clicks[article.getID()] = defaultdict(int)
                self.selections[article.getID()] = defaultdict(int)

        clicks = [self.clicks[article.getID()] for article in possibleActions]
        selections = [self.selections[article.getID()] for article in possibleActions]

        action = possibleActions[np.argmax([(1.0 * x) / y for x, y in zip(clicks, selections)])]

        return action

    def updatePolicy(self, c, a, reward):
        for f, p in enumerate(c.getFeatures()):
            if p is 1:
                if reward is True:
                    self.betaS[a.getID()][f] += 1.0
                else:
                    self.betaF[a.getID()][f] += 1.0
