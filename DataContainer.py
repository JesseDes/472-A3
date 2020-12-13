import pandas
from enum import IntEnum

class DataContainer:
  def __init__(self, name, source):
    df = pandas.read_csv(source, header = None, sep='\t')
    self.parsedData = []    
    for index, rows in df.iterrows():
        if rows[0] == 'tweet_id':
            continue
        
        self.parsedData.append(rows.tolist()[:3])

    self.name = name


class DataContainerTypes(IntEnum):
  ID = 0
  TEXT = 1
  CLASS = 2
  
