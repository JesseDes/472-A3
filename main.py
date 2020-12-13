from DataContainer import DataContainer, DataContainerTypes
from NB import NaivesBayes
from persoMath import divide

testName = "NB-BOW-FV"
trainingData =  DataContainer("name",'./covid_training.tsv')
bayesModel = NaivesBayes(name=testName, filtering=True)
bayesModel.train(trainingData.parsedData)
testingData = DataContainer("name", './covid_test_public.tsv')

predictions = bayesModel.predict(testingData.parsedData)

matches = [0,0]
class_yes_results = {'TP': 0, 'FP':0, 'FN':0}
class_no_results = {'TP': 0, 'FP':0, 'FN':0}
for index, prediction in enumerate(predictions):
    matches[prediction == testingData.parsedData[index][DataContainerTypes.CLASS]] += 1
    if prediction == testingData.parsedData[index][DataContainerTypes.CLASS]:
        if prediction == 'yes':
            class_yes_results['TP'] +=1
        else:
            class_no_results['TP'] +=1
    else:
        if prediction == 'yes':
            class_yes_results['FP'] += 1
            class_no_results['FN'] += 1
        else:
            class_yes_results['FN'] += 1
            class_no_results['FP'] += 1

accuracy = divide(matches[1], matches[0] + matches[1])
precision_yes = divide(class_yes_results['TP'] , (class_yes_results['TP'] + class_yes_results['FP']))
recall_yes = divide (class_yes_results['TP'] , (class_yes_results['TP'] + class_yes_results['FN']))

precision_no = divide(class_no_results['TP'], (class_no_results['TP'] + class_no_results['FP']))
recall_no = divide(class_no_results['TP'], (class_no_results['TP'] + class_no_results['FN']))

f1_yes = divide(2* precision_yes * recall_yes, (precision_yes + recall_yes))
f1_no = divide(2 * precision_no * recall_no, (precision_no + recall_no))

f = open('./output/eval_' + testName + ".txt", "w")
f.write(str(accuracy) + "\n" + str(precision_yes) + "  " + str(precision_no) + "\n" + str(recall_yes) + "  " + str(recall_no) + "\n" + str(f1_yes) + "  " + str(f1_no))
f.close()

