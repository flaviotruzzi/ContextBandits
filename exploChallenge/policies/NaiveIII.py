__author__ = 'ftruzzi'

# Naive3 solution based on "Linear Bayes Policy for Learning in Contextual-Bandits"
# but not using the pursuit method of calculation

import numpy as np
import random as rn
from collections import defaultdict
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy


def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rn.choice(indices)


class Naive3(ContextualBanditPolicy):
    def __init__(self):
        self.clicksPerFeature = {}
        self.selectionsPerFeature = {}
        self.used = {}
        self.t = 0
        self.epsilon = 0.15
        self.overallFeatureClicks = np.zeros(136)
        self.overallFeatureSelections = np.zeros(136)
        self.clicks = {}
        self.selections = {}
        self.pai = {}

    def getActionToPerform(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.pai:
                self.selections[article.getID()] = np.zeros(136)
                self.pai[article.getID()] = np.ones(136)
                # self.clicksPerFeature[article.getID()] = np.zeros(136)
                # self.selectionsPerFeature[article.getID()] = np.zeros(136)
                # self.selections[article.getID()] = 1
                # self.clicks[article.getID()] = 1
                self.used[article.getID()] = 0

        k = [np.sum(self.pai[a.getID()]) for a in possibleActions]
        choice = possibleActions[np.argmax(k)]
        #print k
        # self.used[choice.getID()] += 1
        # self.t += 1
        # if (self.t % 10000) == 0:
        #     print self.used.values()
        return choice

    def updatePolicy(self, c, a, reward):
        features = np.zeros(136)
        for f, p in enumerate(c.getFeatures()):
            features[f] = p
            self.selections[a.getID()][f] += p

        self.pai[a.getID()] += features * (float(reward) - self.pai[a.getID()]) / (1.0 + self.selections[a.getID()])