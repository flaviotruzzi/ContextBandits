#-------------------------------------------------------------------------------
# Copyright (c) 2012 Jose Antonio Martin H..
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Public License v3.0
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/gpl.html
# 
# Contributors:
#     Jose Antonio Martin H. - Translation to Python from Java
#     Flavio Sales Truzzi - Minor modifications.
#-------------------------------------------------------------------------------

from exploChallenge.logs.LogLineGenerator import LogLineGenerator
from exploChallenge.logs.LogLine import LogLine
import random


class RandomGenerator(LogLineGenerator):

    def RandomGenerator(self, nbOfLines, nbOfContexts, nbOfActions):
        self.nbOfLinesRemaining = nbOfLines
        self.nbOfContexts = nbOfContexts
        self.nbOfActions = nbOfActions
        #random = new Random()
        self.possibleActions = range(nbOfActions)

    def setProbas(self, probas):
        self.probas = probas

    def generateLogLine(self):
        ll = LogLine()
        ll.setContext(random.randint(0, self.nbOfContexts - 1))
        ll.setAction(random.randint(0, self.nbOfActions - 1))
        p = 0.5
        if self.probas is not None:
            p = self.probas[(ll.getContext(), ll.getAction())]

        ll.setReward(random.random() < p)
        self.nbOfLinesRemaining -= 1
        return ll

    def getPossibleActions(self):
        return self.possibleActions

    def hasNext(self):
        return self.nbOfLinesRemaining > 0

