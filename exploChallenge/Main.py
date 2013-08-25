# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Jose Antonio Martin H..
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Public License v3.0
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/gpl.html
#
# Contributors:
#     Jose Antonio Martin H. - Translation to Python from Java
#     Jeremie Mary - very minor adaptation for the challenge
#     Flavio Sales Truzzi - Changed the to a less memory consuming
#                           and pythonic version
#-------------------------------------------------------------------------------
#package exploChallenge;

#import java.io.FileNotFoundException;
import sys
import time

from exploChallenge.eval.Evaluator import Evaluator
from exploChallenge.eval.MyEvaluationPolicy import MyEvaluationPolicy
from exploChallenge.logs.FromFileLogLineGenerator import FromFileLogLineGenerator
from exploChallenge.logs.YahooLogLineReader2 import YahooLogLineReader2
from policies.NaiveBayesII import NaiveBayes

class Main:
    """
     * @param argv: a Python list containing the command line args
     * a usual, argv[0] is the full path name of this program.
     * @throws FileNotFoundException -> IOError?
    """

    def main(self, argv=sys.argv):

        currentTimeMillis = lambda: int(round(time.time() * 1000))
        t = currentTimeMillis()
        reader = None

        try:
            reader = YahooLogLineReader2(
                "/home/ftruzzi/Downloads/yahoodata/ydata-fp-td-clicks-v2_0.201110", 2, 16, 136)
            logStep = 10000
        except:
            reader = YahooLogLineReader2("../yahooTest.txt", 136)
            logStep = 1

        generator = FromFileLogLineGenerator(reader)
        self.policy = NaiveBayes()

        evalPolicy = MyEvaluationPolicy(sys.stdout, logStep, 0)

        ev = Evaluator(generator, evalPolicy, self.policy)

        value = ev.runEvaluation()

        if logStep == 1:
            print("final result => " + str(value))
            print(str(currentTimeMillis() - t) + " ms" + "\n")


if __name__ == '__main__':
    app = Main()
    app.main(sys.argv)
