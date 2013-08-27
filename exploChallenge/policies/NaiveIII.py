__author__ = 'ftruzzi'

# Naive3 solution based on "Linear Bayes Policy for Learning in Contextual-Bandits"
# but not using the pursuit method of calculation

import numpy as np
from collections import defaultdict
from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy


class Naive3(ContextualBanditPolicy):
    def __init__(self):
        self.clicksPerFeature = {}
        self.selectionsPerFeature = {}
        self.t = 0

    def getActionToPerform(self, visitor, possibleActions):

        for article in possibleActions:
            if article.getID() not in self.clicksPerFeature:
                self.clicksPerFeature[article.getID()] = defaultdict(int)
                self.selectionsPerFeature[article.getID()] = defaultdict(int)

        # indices = []
        # for article in possibleActions:
        #     Pai = 0
        #     for feature in xrange(136):
        #         clicks = self.clicksPerFeature[article.getID()][feature]
        #         selections = self.selectionsPerFeature[article.getID()][feature]
        #         if selections is not 0:
        #             Pai += 1.0 * clicks / selections
        #     indices.append(Pai)

        clicks = [np.asarray(self.clicksPerFeature[article.getID()].values())[1:] for article in possibleActions]
        selections = [np.asarray(self.selectionsPerFeature[article.getID()].values())[1:] for article in
                      possibleActions]

        nFeatures = [sum((k != 0)) for k in selections]

        indices = [sum(1.0 * clicks[x] / (selections[x] + 1) / (nFeatures[x] + 1)) for x in
                   xrange(len(possibleActions))]
        indices = [1 if x is 0 else x for x in indices]



        # self.t += 1
        # if self.t == 333:
        #     print 'k'
        # print self.t, indices

        action = possibleActions[np.argmax(indices)]

        return action

    def updatePolicy(self, c, a, reward):
        for f, p in enumerate(c.getFeatures()):
            self.clicksPerFeature[a.getID()][f] += p * int(reward)
            self.selectionsPerFeature[a.getID()][f] += p
