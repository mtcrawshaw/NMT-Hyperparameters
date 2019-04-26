import json

LOG_PATH = './regularLog.txt'
OUTFILE_PATH = './regularResults.json'

def getTrainingAccuracy(line):
    metrics = line.split(';')
    entries = metrics[1].split(' ')
    trainingAccuracy = float(entries[-1])

    return trainingAccuracy

def getValidationAccuracy(line):
    entries = line.split(" ")
    validationAccuracy = float(entries[-1].strip())

    return validationAccuracy


def main():
    results = {}
    trainingAccuracies = []
    validationAccuracies = []

    # Read in file
    with open(LOG_PATH, 'r') as f:
        lines = f.readlines()

    # Iterate through lines to get validation and training accuracies
    for line in lines:
        # Check if line contains training accuracy
        if 'Step' in line:
            trainingAccuracy = getTrainingAccuracy(line)
            trainingAccuracies.append(trainingAccuracy)

        # Check if line contains validation accuracy
        if 'Validation accuracy' in line:
            validationAccuracy = getValidationAccuracy(line)
            validationAccuracies.append(validationAccuracy)

    results['training'] = list(trainingAccuracies)
    results['validation'] = list(validationAccuracies)

    # Save out file
    with open(OUTFILE_PATH, 'w') as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
