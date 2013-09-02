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
        self.overallFeatureClick = {}
        self.overallFeatureSelection = {}
        self.t = 0

    def getActionToPerform(self, visitor, possibleActions):

        self.t += 1

        for action in possibleActions:
            if action.getID() not in self.articlesS:
                self.articlesS[action.getID()] = 1.0
                self.articlesF[action.getID()] = 1.0
                self.featuresPerArticleS[action.getID()] = np.ones(135)  # The first n
                self.featuresPerArticleF[action.getID()] = np.ones(135)
                #self.overallFeatureClick[action.getID()] = 1.0
                #self.overallFeatureSelection[action.getID()] = 1.0

        alphas = [20 * (self.articlesS[a.getID()] + np.sum(1. / 135 * self.featuresPerArticleS[a.getID()])) for a in possibleActions]
        betas = [20 * (self.articlesF[a.getID()] + np.sum(1. / 135 * self.featuresPerArticleF[a.getID()])) for a in possibleActions]

        params = np.vstack((alphas, betas)).T
        if self.t % 10000 is 0:
            print params, params.T[0, :] / params.T[1, :]
        indices = [np.random.beta(*param) for param in params]

        choice = np.argmax(indices)

        return possibleActions[choice]

    def updatePolicy(self, c, a, reward):

        if reward is True:
            self.articlesS[a.getID()] += 1.0

            for f, p in enumerate(c.getFeatures()):
                if f is not 0:
                    self.featuresPerArticleS[a.getID()][f - 1] += p * 1.0

        else:
            self.articlesF[a.getID()] += 1.0

            for f, p in enumerate(c.getFeatures()):
                if f is not 0:
                    self.featuresPerArticleF[a.getID()][f - 1] += p * 1.0