__author__ = 'ftruzzi'

# My naive bayes approach
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
        self.clicks = {}
        self.selections = {}
        self.clicksPerFeature = {}
        self.selectionsPerFeature = {}

    def getActionToPerform(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.clicks:
                self.clicks[article.getID()] = 1.0 #Optimistic
                self.selections[article.getID()] = 1.0
                self.clicksPerFeature[article.getID()] = np.ones(136)
                self.selectionsPerFeature[article.getID()] = np.ones(136)


        # TODO: change to vectorized or list-comprehension
        indices = []
        for a in possibleActions:
            index = self.clicks[a.getID()] / self.selections[a.getID()]
            for f in visitor.getFeatures():
                index *= self.clicksPerFeature[a.getID()][f] / self.clicksPerFeature[a.getID()][f]
            indices.append(index)

        #k = [np.sum(self.pai[a.getID()]) for a in possibleActions]
        choice = possibleActions[np.argmax(indices)]
        #print k
        # self.used[choice.getID()] += 1
        # self.t += 1
        # if (self.t % 10000) == 0:
        #     print self.used.values()
        return choice

    def updatePolicy(self, c, a, reward):
        self.clicks[a.getID()] += reward
        self.selections[a.getID()] += 1.0
        for f, p in enumerate(c.getFeatures()):
            # Feature is "on" and there is a reward given the article
            self.clicksPerFeature[a.getID()][f] += p * float(reward)
            # Feature is present is given the article
            self.selectionsPerFeature[a.getID()][f] += p

            #self.pai[a.getID()] += features * (float(reward) - self.pai[a.getID()]) / (1.0 + self.selections[a.getID()])