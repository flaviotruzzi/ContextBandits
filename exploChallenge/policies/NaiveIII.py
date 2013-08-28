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

    def getActionToPerform(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.clicksPerFeature:
                self.clicksPerFeature[article.getID()] = np.zeros(136)
                self.selectionsPerFeature[article.getID()] = np.zeros(136)
                self.selections[article.getID()] = 1
                self.clicks[article.getID()] = 1
                self.used[article.getID()] = 0



        if np.random.random() > self.epsilon:

            indices = []
            for a in possibleActions:
                index = 0
                ar = a.getID()
                for i in xrange(136):
                    index += (1.0 * self.clicksPerFeature[ar][i] / (1 + self.selectionsPerFeature[ar][i])) * (
                       1.0 * self.clicks[ar] / (self.selections[ar] + 1)) / (
                                1.0 + self.overallFeatureClicks[i] / (1 + self.overallFeatureSelections[i]))
                indices.append(index)

            #print indices
            choice = possibleActions[rargmax(indices)]
        else:
            choice = rn.choice(possibleActions)

        #choice = rn.choice(possibleActions)
        self.used[choice.getID()] += 1
        if (self.t % 10000) == 0:
            print self.used.values()
        return choice

    def updatePolicy(self, c, a, reward):
        #if reward is True:
        #    print '####################################################################################################'
        for f, p in enumerate(c.getFeatures()):
            self.clicksPerFeature[a.getID()][f] += p * int(reward)
            self.selectionsPerFeature[a.getID()][f] += p
            self.overallFeatureClicks[f] += p * int(reward)
            self.overallFeatureSelections[f] += p
            self.clicks[a.getID()] += p * int(reward)
            self.selections[a.getID()] += p

