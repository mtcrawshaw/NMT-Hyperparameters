import sys
import argparse
import json

from hyperparameter import declareParams
from genetic import initializePopulation, getSurvivorIndices, getNextGeneration
from train import trainMember, copyModels

"""

  main script for hyperparameter optimization of NMT using a genetic algorithm.
  All options for genetic algorithm aren't command line arguments, they are constant
  values in this script (directly below this comment block).

"""

# Parameters for genetic algorithm
POPULATION_SIZE = 6
NUM_WINNERS = 2 # These are the number of top members which are chosen for next generation
NUM_LUCKY = 1 # These are the number of members which are chosen for next generation randomly
NUM_GENERATIONS = 8

LOG_PATH = './log.json'

def main(args):
    params = declareParams()
    population = initializePopulation(POPULATION_SIZE, params)

    generationsLog = []

    # Iterate over generations
    for generation in range(NUM_GENERATIONS):

        generationLog = {}
        generationLog['population'] = list(population)

        # Print current generation info
        print("Generation %d:" % generation)
        for i, member in enumerate(population):
            print("%d: " % i, end="")
            print(member)
        print("")

        # Iterate over population
        accuracies = []
        trainAccuracies = []
        validAccuracies = []
        for i, member in enumerate(population):
            trainAccuracy, validAccuracy = trainMember(generation, i, member,
                    args.trainingSteps, args.batchSize)
            accuracies.append((i, trainAccuracy))
            trainAccuracies.append(trainAccuracy)
            validAccuracies.append(validAccuracy)

        generationLog['trainAccuracies'] = list(trainAccuracies)
        generationLog['validAccuracies'] = list(validAccuracies)
        generationsLog.append(dict(generationLog))

        # Print performance of each population member
        print("Performances:")
        for i, accuracy in accuracies:
            print("%d accuracy: %.5f" % (i, accuracy))
        print("")

        # Get survivors
        survivorIndices = getSurvivorIndices(accuracies, POPULATION_SIZE, NUM_WINNERS, NUM_LUCKY)
        survivors = [population[i] for i in survivorIndices]
        print("Survivors:")
        for survivor in survivors:
            print(survivor)
        
        # Get next generation
        population = getNextGeneration(survivors, POPULATION_SIZE, params)

        # Copy best model to overwrite all other models
        bestIndex = survivorIndices[0]
        copyModels(bestIndex, POPULATION_SIZE)

        print("\n--------------------\n")

    # Write out log
    log = {}
    log['generations'] = list(generationsLog)
    with open(LOG_PATH, 'w') as f:
        f.write(json.dumps(log, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic algorithm to \
    optimize hyperparameters of NMT training')

    parser.add_argument('--batchSize', type=int, default=192)
    parser.add_argument('--trainingSteps', type=int, default=400)
    args = parser.parse_args()

    main(args)
