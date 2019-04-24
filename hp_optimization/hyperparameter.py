import random

smallMutate = lambda x, e : x * (1. + (2. * e - 1.) / 10.)
bigMutate = lambda x, e : x * (1. + 9. * (2. * e - 1.) / 10.)

class HyperParameter:

    def __init__(self, name, defaultValue, minValue, maxValue, mutate):
        self.name = name
        self.defaultValue = defaultValue
        self.minValue = minValue
        self.maxValue = maxValue

        # self.mutate is a function which takes as input two floats,
        # x and e. x is a value to be mutated, e is assumed to be any
        # number between zero and one. self.mutate returns a mutated
        # value of x using e.
        self.mutate = mutate

    def mutateWithinBounds(self, value):
        newValue = self.mutate(value, random.random())

        if newValue > self.maxValue:
            newValue = (value + self.maxValue) / 2.0
        elif newValue < self.minValue:
            newValue = (value + self.minValue) / 2.0

        return newValue

    def getInitialValue(self):
        return self.mutateWithinBounds(self.defaultValue)

def declareParams():
    params = {}

    params['dropout'] = HyperParameter(
        'dropout',
        0.3,
        0,
        1,
        smallMutate
    )
    params['learning_rate'] = HyperParameter(
        'learning_rate',
        0.001,
        1e-9,
        1,
        bigMutate
    )
    params['adam_beta1'] = HyperParameter(
        'adam_beta1',
        0.9,
        0,
        1,
        smallMutate
    )
    params['adam_beta2'] = HyperParameter(
        'adam_beta2',
        0.999,
        0,
        1,
        smallMutate
    )
    params['label_smoothing'] = HyperParameter(
        'label_smoothing',
        0.1,
        0,
        1,
        smallMutate
    )

    return params
