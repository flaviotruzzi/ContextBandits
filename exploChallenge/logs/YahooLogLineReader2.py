__author__ = 'ftruzzi'

from yahoo.YahooVisitor import YahooVisitor
from yahoo.YahooArticle import YahooArticle
from yahoo.YahooLogLine import YahooLogLine


class YahooLogLineReader2:
    """ Works like the Original YahooLogLineReader, but using generators.
    """

    def __init__(self, a, b, c=None, d=None):
        self.files = []
        if c is None and d is None:
            self.__init1__(a, b)
        else:
            self.__init2__(a, b, c, d)

    def __init2__(self, prefix, firstFile, lastFile, nbOfFeatures):
        self.nbOfFeatures = nbOfFeatures
        self.currentFile = firstFile - 1
        self.lastFile = lastFile
        self.filePrefix = prefix

        while firstFile <= lastFile:
            self.files.append(self.filePrefix + "%02d" % firstFile)
            firstFile += 1

    def __init1__(self, filePath, nbOfFeatures):
        self.nbOfFeatures = nbOfFeatures
        self.filePath = filePath
        self.currentFile = 0
        self.lastFile = 0
        self.files.append(self.filePath)

    def read(self):
        for filename in self.files:
            with open(filename) as f:
                print filename
                for line in f:
                    l = line.strip().split(' |')
                    timestamp, ad, click = l[0].split()
                    ad = ad[3:]
                    features = [0 for _ in xrange(self.nbOfFeatures)]
                    for i in l[1][5:].split():
                        features[int(i) - 1] = 1
                    self.possibleActions = [YahooArticle(int(PA[3:])) for PA in l[2:]]
                    visitor = YahooVisitor(timestamp, features)
                    article = YahooArticle(int(ad))
                    logLine = YahooLogLine(visitor, article, click == '1')

                    yield logLine

    def getPossibleActions(self):
        return self.possibleActions

