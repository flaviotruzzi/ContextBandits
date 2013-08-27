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
        self.epsilon = 0.05
        self.overallFeatureClicks = np.zeros(136)
        self.overallFeatureSelections = np.zeros(136)
        self.clicks = {}
        self.selections = {}

    def getActionToPerform(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.clicksPerFeature:
                self.clicksPerFeature[article.getID()] = np.zeros(136)
                self.selectionsPerFeature[article.getID()] = np.zeros(136)
                self.selections[article.getID()] = 0
                self.clicks[article.getID()] = 0
                self.used[article.getID()] = 0

        if np.random.random() > self.epsilon:

            print [sum((self.clicksPerFeature[a.getID()] / (1 + self.selections[a.getID()])) / ((self.overallFeatureClicks / (1 + self.overallFeatureSelections)))) *
                   (self.clicks[a.getID()] / (1 + self.selections[a.getID()])) for a in possibleActions]

            #selections = [self.selectionsPerFeature[article.getID()] for article in possibleActions]
            #nbFeatures = np.count_nonzero(selections) + 1
            #indices = [1 + sum((1.0 * clicks[x][1:]) / (selections[x][1:] + 1)) / nbFeatures for x in xrange(len(possibleActions))]
            #action = possibleActions[np.argmax(indices)]
            #action = possibleActions[rargmax(indices)]
            #print self.t, indices
            #print self.used.values()
            #else:
            #    action = rn.choice(possibleActions)

            self.t += 1

            #self.used[action.getID()] += 1

        return rn.choice(possibleActions)

    def updatePolicy(self, c, a, reward):
        #if reward is True:
        #    print '####################################################################################################'
        for f, p in enumerate(c.getFeatures()):
            self.clicksPerFeature[a.getID()][f] += p * int(reward)
            self.selectionsPerFeature[a.getID()][f] += p
            self.overallFeatureClicks[f] += p * int(reward)
            self.overallFeatureSelections[f] += p

