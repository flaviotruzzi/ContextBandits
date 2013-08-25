#-------------------------------------------------------------------------------
# Copyright (c) 2012 Jose Antonio Martin H..
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Public License v3.0
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/gpl.html
# 
# Contributors:
#     Jose Antonio Martin H. - Translation to Python from Java
#-------------------------------------------------------------------------------
from exploChallenge.eval.IllegalChoiceOfArticleException import IllegalChoiceOfArticleException


class Evaluator:

    def __init__(self, generator, evalPolicy, policy, linesToSkip=0):
        self.generator = generator
        self.evalPolicy = evalPolicy
        self.policy = policy
        self.linesToSkip = linesToSkip

    def runEvaluation(self):
        #while self.generator.hasNext():
        #    logLine = self.generator.generateLogLine()

        for logLine in self.generator.generateLogLine():
#            print logLine.action.getID(), logLine.reward
            if self.linesToSkip > 0:
                self.linesToSkip -= 1
                continue

            a = self.policy.getActionToPerform(logLine.getContext(), self.generator.getPossibleActions())
            if a not in self.generator.getPossibleActions():
                raise IllegalChoiceOfArticleException
            self.evalPolicy.evaluate(logLine, a)
            if a == logLine.getAction():
                self.policy.updatePolicy(logLine.getContext(), logLine.getAction(), logLine.getReward())

        return self.evalPolicy.getResult()



