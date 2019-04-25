import os
import subprocess
import random

projectRoot = os.path.dirname(os.path.dirname(__file__))
trainPath = os.path.join(projectRoot, 'train.py')
dataPath = os.path.join(projectRoot, 'data', 'demo')
modelsPath = os.path.join(projectRoot, 'hp_optimization', 'models')

def trainMember(i, member, trainingSteps, big=True):

    batch_size = 256 if big else 8
    trainingSteps = trainingSteps if big else 50

    command = "python3 %s" % trainPath
    command += " -data %s" % dataPath
    command += " -save_model %s" % os.path.join(modelsPath, 'genetic_%d.pt' % i)
    command += " -gpu_ranks 0"
    command += " -train_from %s" % os.path.join(modelsPath, 'genetic_%d.pt' % i)
    command += " -batch_size %d" % batch_size
    command += " -valid_batch_size %d" % batch_size
    command += " -train_steps %d" % trainingSteps
    command += " -valid_steps %d" % trainingSteps
    command += " -optim adam"
    command += " -learning_rate_decay 0"
    command += " -log_file train.log"

    for paramName in member:
        command += " -%s %f" % (paramName, member[paramName])

    subprocess.call(command, shell=True)

    return random.random(), random.random(), random.random(), random.random()

def copyModels(i):
    pass
