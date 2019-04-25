import sys

from hyperparameter import declareParams
from genetic import initializePopulation, getSurvivorIndices, getNextGeneration
from train import trainMember, copyModels

"""

  main script for hyperparameter optimization of NMT using a genetic algorithm.
  All options for genetic algorithm aren't command line arguments, they are constant
  values in this script (directly below this comment block). The only argument to
  main.py is an optional "--small", which will run the training with a small batch
  size and small number of training steps, for use on a smaller GPU/testing purposes.

"""

# Parameters for genetic algorithm
POPULATION_SIZE = 6
NUM_WINNERS = 2 # These are the number of top members which are chosen for next generation
NUM_LUCKY = 1 # These are the number of members which are chosen for next generation randomly
NUM_GENERATIONS = 5
TRAINING_STEPS = 100 # Training steps per member per generation

def main(big=True):
    params = declareParams()
    population = initializePopulation(POPULATION_SIZE, params)

    # Iterate over generations
    for generation in range(NUM_GENERATIONS):

        # Print current generation info
        print("\n--------------------\n")
        print("Generation %d:" % generation)
        for i, member in enumerate(population):
            print("%d: " % i, end="")
            print(member)
        print("")

        # Iterate over population
        losses = []
        print("Performances:")
        for i, member in enumerate(population):
            trainLoss, trainAcc, validLoss, validAcc = trainMember(i, member, TRAINING_STEPS, big)
            losses.append((i, trainLoss))
            print("%d loss: %.5f" % (i, trainLoss))
        print("")

        # Get survivors
        survivorIndices = getSurvivorIndices(losses, POPULATION_SIZE, NUM_WINNERS, NUM_LUCKY)
        survivors = [population[i] for i in survivorIndices]
        print("Survivors:")
        for survivor in survivors:
            print(survivor)
        
        # Get next generation
        population = getNextGeneration(survivors, POPULATION_SIZE, params)

        # Copy best model to overwrite all other models
        bestIndex = survivorIndices[0]
        copyModels(bestIndex)

        print("\n--------------------\n")
        

if __name__ == "__main__":
    big = True
    if len(sys.argv) > 1 and sys.argv[1] == '--small':
        big = False
    main(big)
