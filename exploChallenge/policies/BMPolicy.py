__author__ = 'ftruzzi'

from collections import defaultdict
import numpy as np


class Beta:
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta

    def __mul__(self, other):
        return Beta(self.alpha + other.alpha, self.beta + other.beta)

    def __repr__(self):
        return '(' + repr(self.alpha) + ',' + repr(self.beta) + ')'

    def sample(self):
        return np.random.beta(self.alpha, self.beta)


class BMPolicy:
    """Beta Mixture Policy"""

    def __init__(self):
        self.articlesS = {}
        self.articlesF = {}
        self.featuresPerArticleS = {}
        self.featuresPerArticleF = {}
        self.overallFeatureClick = np.ones(135)
        self.overallFeatureSelection = np.ones(135)
        self.lifetime = defaultdict(int)
        self.t = 0

    def getActionToPerform(self, visitor, possibleActions):

        """

        @param visitor:
        @param possibleActions:
        @return:
        """
        self.t += 1

        for action in possibleActions:
            self.lifetime[action.getID()] += 1
            if action.getID() not in self.articlesS:
                self.articlesS[action.getID()] = 1.0
                self.articlesF[action.getID()] = 1.0
                self.featuresPerArticleS[action.getID()] = np.ones(135)  # The first n
                self.featuresPerArticleF[action.getID()] = np.ones(135)

        indices = []
        for article in possibleActions:
            fIndex = 0.0
            for f, p in enumerate(visitor.getFeatures()):
                if p is 1 and f is not 0:
                    fIndex += np.random.beta( self.featuresPerArticleS[article.getID()][f - 1],
                                              self.featuresPerArticleF[article.getID()][f - 1])
            fIndex *= np.random.beta( self.articlesS[article.getID()], self.articlesF[article.getID()])
            indices.append(fIndex)

        choice = np.argmax(indices)

        return possibleActions[choice]

    def updatePolicy(self, c, a, reward):

        if reward is True:
            self.articlesS[a.getID()] += 1.0

            for f, p in enumerate(c.getFeatures()):
                if f is not 0:
                    self.featuresPerArticleS[a.getID()][f - 1] += p * 1.0
                    self.overallFeatureClick[f - 1] += p * 1.0
        else:
            self.articlesF[a.getID()] += 1.0

            for f, p in enumerate(c.getFeatures()):
                if f is not 0:
                    self.featuresPerArticleF[a.getID()][f - 1] += p * 1.0
                    self.overallFeatureSelection[f - 1] += p * 1.0