__author__ = 'ftruzzi'
import numpy as np


class GMPolicy():
    """"""

    def __init__(self):
        """Constructor for GMPolicy"""
        self.dimension = 136   # Number of features

        self.B = {}  # np.identity(self.dimension)
        self.Binv = {}
        self.emu = {}  # np.zeros((self.dimension, 1))
        self.f = {}  # np.zeros((self.dimension, 1))

        R = 5
        delta = .05
        d = 136
        self.asd = np.zeros(35)
        self.v2 = R * R * (np.log(1 / delta) * 24. / d)


    def getActionToPerform(self, visitor, possibleActions):

        for action in possibleActions:
            if action.getID() not in self.emu:
                self.emu[action.getID()] = np.zeros((self.dimension, 1))
                self.f[action.getID()] = np.zeros((self.dimension, 1))
                self.B[action.getID()] = np.identity(self.dimension)
                self.Binv[action.getID()] = np.identity(self.dimension)

        features = np.zeros((self.dimension, 1))
        features[visitor.getFeatures()] = 1.0

        # sample tmu from N(emu, v2B-1)
        tmu = {action.getID(): np.random.multivariate_normal(np.squeeze(self.emu[action.getID()]),
            self.v2 * self.Binv[action.getID()]) for action in possibleActions}

        # play arm that maximize bTtmu
        e = [np.dot(np.squeeze(features), tmu[action.getID()].T) for action in possibleActions]
        #choice = np.argmax(e)
        #self.asd[choice] += 1
        #print self.asd

        return possibleActions[np.argmax(e)]


    def updatePolicy(self, c, a, reward):
        features = np.zeros((self.dimension, 1))
        features[c.getFeatures()] = 1.0

        self.B[a.getID()] += np.dot(features, features.T)
        self.Binv[a.getID()] = np.linalg.inv(self.B[a.getID()])
        self.f[a.getID()] += features * float(reward)

        self.emu[a.getID()] = np.dot(np.linalg.inv(self.B[a.getID()]), self.f[a.getID()])  # Estimated mu

