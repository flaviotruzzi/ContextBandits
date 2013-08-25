#-------------------------------------------------------------------------------
# Copyright (c) 2012 Jose Antonio Martin H..
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Public License v3.0
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/gpl.html
# 
# Contributors:
#     Jose Antonio Martin H. - Translation to Python from Java
#     Flavio Sales Truzzi - Changed to work with YahooLogLineReader2
#-------------------------------------------------------------------------------
#package exploChallenge.logs;

from exploChallenge.logs.LogLineGenerator import LogLineGenerator


class FromFileLogLineGenerator(LogLineGenerator):

    reader = None

    def __init__(self, reader):
        self.reader = reader

    def generateLogLine(self):
        return self.reader.read()

    def hasNext(self):
        try:
            return self.reader.hasNext()
        except IOError as (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
            #catch (IOException e) {
            #e.printStackTrace();
            raise

    def getPossibleActions(self):
        return self.reader.getPossibleActions()


