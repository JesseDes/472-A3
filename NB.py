import math
from DataContainer import DataContainer, DataContainerTypes
from enum import IntEnum
from datetime import datetime
import string  


class NaivesBayes:
  def __init__(self, name='', smoothing = 0.01, filtering=False):
      self.smoothing = smoothing
      self.filtering = filtering
      self.trainingData = {}
      self.classFreq = [0,0]
      self.name = name

  def train(self, trainingData):
      for datum in trainingData:
        for word in datum[DataContainerTypes.TEXT].split():
            word = self.__normalizeWord__(word)
            if not word in self.trainingData:
                self.trainingData[word] = [0,0]
            
            self.trainingData[word][datum[DataContainerTypes.CLASS] == 'yes'] += 1
        
        self.classFreq[datum[DataContainerTypes.CLASS] == 'yes'] += 1
                
      if self.filtering == True:
          deletionList = []
          for datum in self.trainingData:
              if not (self.trainingData[datum][0] + self.trainingData[datum][1]) > 1:
                  deletionList.append(datum)
          for datum in deletionList:
              del self.trainingData[datum]
          del deletionList

  def predict(self, testData):
      totalClassInstances = self.classFreq[0] + self.classFreq[1]
      f = open('./output/trace_' + self.name + ".txt", "w")
      f.write("") 
      f.close()
      predictions = []
      for datum in testData:
          predictions.append(self.__testDatum__(datum, totalClassInstances))

      return predictions
  
  def __testDatum__(self, datum, totalClassInstances):
      highestScore = None
      for index, freqOfClass in enumerate(self.classFreq):
        currentScore = math.log10(freqOfClass / totalClassInstances) 
        for word in datum[DataContainerTypes.TEXT].split():
          if word in self.trainingData and self.trainingData[word][index] > 0:
            currentScore += math.log10((self.trainingData[word][index] + self.smoothing) / (freqOfClass + (len(self.trainingData) * self.smoothing)))

        if highestScore == None or currentScore > highestScore:
            bestClass = index
            highestScore = currentScore

      self.__trace__(datum, bestClass, highestScore)
      if bestClass:
          return 'yes'
      else:
          return 'no'

  def __normalizeWord__(self, word):
      newWord = word.lower()
      newWord = newWord.translate(str.maketrans('', '', string.punctuation))
      return newWord

  def __trace__(self , datum, bestClass, score):
      if bestClass:
          bestClassString = "yes"
      else:
          bestClassString = "no"

      if bestClassString == datum[DataContainerTypes.CLASS]:
          matchingClass = "correct"
      else:
          matchingClass = "wrong"

      formattedScore = "{:e}".format(score)

      f = open('./output/trace_' + self.name + ".txt", "a")
      f.write(str(datum[DataContainerTypes.ID]) + "  " + bestClassString  + "  " + formattedScore + "  " + datum[DataContainerTypes.CLASS] + "  " + matchingClass + "\n" )
      f.close()
