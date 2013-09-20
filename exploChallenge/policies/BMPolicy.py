__author__ = 'ftruzzi'

from collections import defaultdict
import numpy as np
import pickle

class BMPolicy:
    """Beta Mixture Policy"""

    def __init__(self):
        self.articlesS = {}
        self.articlesF = {}
        self.featuresPerArticleS = {}
        self.featuresPerArticleF = {}
        self.overallFeatureClick = np.ones(135)
        self.overallFeatureSelection = np.ones(135)
        self.choices = defaultdict(int)
        self.t = 0
        try:
            with open('learnt.npz', 'r') as f:
                self.articlesS, self.articlesF, self.featuresPerArticleS, self.featuresPerArticleF = pickle.load(f)
                print "Starting with knowledge"
        except IOError:
            print 'Oh dear. From Scratch'

    def getActionToPerform(self, visitor, possibleActions):

        """

        @param visitor:
        @param possibleActions:
        @return:
        """
        self.t += 1

        for action in possibleActions:
            if action.getID() not in self.articlesS:
                self.articlesS[action.getID()] = 1.0
                self.articlesF[action.getID()] = 1.0
                self.featuresPerArticleS[action.getID()] = np.ones(135)
                self.featuresPerArticleF[action.getID()] = np.ones(135)

        indices = []
        for article in possibleActions:
            fIndex = 0.0
            for f, p in enumerate(visitor.getFeatures()):
                if p is 1 and f is not 0:
                    fIndex += np.random.beta(self.featuresPerArticleS[article.getID()][f - 1],
                                             self.featuresPerArticleF[article.getID()][f - 1])
            fIndex *= np.random.beta(self.articlesS[article.getID()], self.articlesF[article.getID()])
            indices.append(fIndex)

        mmean = [self.articlesS[a.getID()] / (self.articlesS[a.getID()] + self.articlesF[a.getID()]) for a in possibleActions]

        # if 2 * max(indices) > max(mmean):
        #     choice = np.argmax(indices)
        # else:
        #     choice = np.argmax(mmean)


        if self.t % 1200000 == 0:
            f = open('learnt.npz', 'w')
            pickle.dump([self.articlesS, self.articlesF, self.featuresPerArticleS, self.featuresPerArticleF], f)
            f.close()
            print "Saved."

        # self.choices[possibleActions[choice].getID()] += 1
        # if self.t % 10000 == 0:
        #     print indices, choice, max(indices)
        #     print '#', np.asarray(self.articlesS.values()) / (np.asarray(self.articlesF.values()) + np.asarray(self.articlesS.values())), max(np.asarray(self.articlesS.values()) / (np.asarray(self.articlesF.values()) + np.asarray(self.articlesS.values()))), np.argmax(np.asarray(self.articlesS.values()) / (np.asarray(self.articlesF.values()) + np.asarray(self.articlesS.values())))
        #     print '$', self.choices.values(), np.argmax(self.choices.values())

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