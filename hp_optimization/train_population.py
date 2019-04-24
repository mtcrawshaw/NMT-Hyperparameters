from hyperparameter import declareParams
from genetic import initializePopulation, trainMember, getSurvivors, getNextGeneration

# Parameters for genetic algorithm
POPULATION_SIZE = 6
NUM_WINNERS = 2 # These are the number of top members which are chosen for next generation
NUM_LUCKY = 1 # These are the number of members which are chosen for next generation randomly
NUM_GENERATIONS = 5
TRAINING_STEPS = 100 # Training steps per member per generation

def main():
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
            trainLoss, trainAcc, validLoss, validAcc = trainMember(i, member, TRAINING_STEPS)
            losses.append((i, trainLoss))
            print("%d loss: %.5f" % (i, trainLoss))
        print("")

        # Get survivors
        survivors = getSurvivors(losses, population, POPULATION_SIZE, NUM_WINNERS, NUM_LUCKY)
        print("Survivors:")
        for survivor in survivors:
            print(survivor)
        
        # Get next generation
        population = getNextGeneration(survivors, POPULATION_SIZE, params)
        

if __name__ == "__main__":
    main()
