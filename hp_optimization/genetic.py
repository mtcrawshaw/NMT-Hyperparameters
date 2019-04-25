import random

def initializePopulation(populationSize, params):
    population = []

    # Fill population
    while len(population) < populationSize:
        member = {}

        # Fill value of each parameter
        for paramName, param in params.items():
            member[paramName] = param.getInitialValue()

        population.append(dict(member))

    return population

def getSurvivorIndices(losses, populationSize, numWinners, numLucky):

    # Sort members by loss
    losses = sorted(losses, key = lambda lossTuple : lossTuple[1])
    
    # Choose survivors, first winners, then randomly
    survivorIndices = [lossTuple[0] for lossTuple in losses[:numWinners]]
    while len(survivorIndices) < (numWinners + numLucky):
        candidateSurvivorIndex = random.randint(0, populationSize - 1)

        if candidateSurvivorIndex not in survivorIndices:
            survivorIndices.append(candidateSurvivorIndex)

    return survivorIndices

def getChild(survivor, params):
    child = {}

    for paramName, param in params.items():
        child[paramName] = param.mutateWithinBounds(survivor[paramName])

    return child

def getNextGeneration(survivors, populationSize, params):
    childrenPerSurvivor = populationSize // len(survivors)
    nextGeneration = []

    # Get equal number of children from each survivor
    for survivor in survivors:
        for i in range(childrenPerSurvivor):
            child = getChild(survivor, params)
            nextGeneration.append(dict(child))

    # Fill remaining population slots with children of random survivors
    while len(nextGeneration) < populationSize:
        child = getChild(random.choice(survivors), params)
        nextGeneration.append(dict(child))

    return nextGeneration
