import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import json

PLOTS_DIR = '../plots'
GENETIC_PATH = '../logs/geneticResults.json'
REGULAR_PATH = '../logs/regularResults.json'
PARAM_NAMES = ['dropout', 'learning_rate', 'adam_beta1', 'adam_beta2',
        'label_smoothing']

GENETIC_STEP_SIZE = 400
REGULAR_STEP_SIZE = 144

def createAccuracyPlot(geneticResults, regularResults, training=True):

    # Build genetic accuracy list
    geneticSteps = []
    geneticAccuracies = []
    for i, generation in enumerate(geneticResults['generations']):

        populationSize = len(generation['trainAccuracies'])
        geneticSteps += [(i * populationSize + j) * GENETIC_STEP_SIZE
                for j in range(1, populationSize + 1)]
        if training:
            geneticAccuracies += generation['trainAccuracies']
        else:
            geneticAccuracies += generation['validAccuracies']

    # Build regular accuracy list
    regularAccuracies = regularResults['training'] if training else \
            regularResults['validation']
    regularSteps = [i * REGULAR_STEP_SIZE for i in 
            range(len(regularAccuracies))]

    # Create plot
    geneticX = np.array(geneticSteps)
    geneticY = np.array(geneticAccuracies)
    regularX = np.array(regularSteps)
    regularY = np.array(regularAccuracies)

    plt.plot(geneticX, geneticY)
    plt.plot(regularX, regularY)
    plt.legend(['genetic', 'baseline'])

    plotName = 'genetic'
    plotName += 'Training' if training else 'Validation'
    plotName += 'Accuracy.png'
    plt.savefig(os.path.join(PLOTS_DIR, plotName))

    # Clear plot
    plt.clf()


def createParamPlot(geneticResults, paramName):

    # Build parameter value list
    steps = []
    parameterValues = []
    for i, generation in enumerate(geneticResults['generations']):

        populationSize = len(generation['trainAccuracies'])
        steps += [(i * populationSize + j) * GENETIC_STEP_SIZE
                for j in range(1, populationSize + 1)]
        parameterValues += [generation['population'][j][paramName] for j in
                range(populationSize)]

    # Create plot
    x = np.array(steps)
    y = np.array(parameterValues)

    plt.plot(x, y)
    plt.savefig(os.path.join(PLOTS_DIR, 'genetic_%s.png' % paramName))

    # Clear plot
    plt.clf()


def main():

    # Load in results
    with open(GENETIC_PATH, 'r') as f:
        geneticResults = json.load(f)
    with open(REGULAR_PATH, 'r') as f:
        regularResults = json.load(f)

    # Create plots
    createAccuracyPlot(geneticResults, regularResults, training=True)
    createAccuracyPlot(geneticResults, regularResults, training=False)
    for paramName in PARAM_NAMES:
        createParamPlot(geneticResults, paramName)


if __name__ == "__main__":
    main()
