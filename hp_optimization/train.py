import os
import subprocess
from shutil import copyfile

TEMP_LOG_PATH = 'temp_log.txt'

projectRoot = os.path.dirname(os.path.dirname(__file__))
trainPath = os.path.join(projectRoot, 'train.py')
dataPath = os.path.join(projectRoot, 'data', 'demo')
modelsPath = os.path.join(projectRoot, 'hp_optimization', 'models')

def getAccuracyFromLog():
    with open(TEMP_LOG_PATH, 'r') as f:
        lines = f.readlines()

    trainingAccuracy = None
    for line in lines:
        if 'Step' in line:
            metrics = line.split(';')
            entries = metrics[1].split(' ')
            trainingAccuracy = float(entries[-1])

        if 'Validation accuracy' in line:
            entries = line.split(" ")
            validationAccuracy = float(entries[-1].strip())

    if trainingAccuracy is None:
        raise RuntimeError('Could not read training accuracy from %s' %
                TEMP_LOG_PATH)
    if validationAccuracy is None:
        raise RuntimeError('Could not read validation accuracy from %s' %
                TEMP_LOG_PATH)

    return trainingAccuracy, validationAccuracy


def trainMember(generation, i, member, trainingSteps, batchSize):

    totalTrainSteps = trainingSteps * (generation + 1)

    command = "python3 %s" % trainPath
    command += " -data %s" % dataPath
    command += " -save_model %s" % os.path.join(modelsPath, 'genetic_%d' % i)
    command += " -gpu_ranks 0"
    command += " -train_from %s" % os.path.join(modelsPath, 'genetic_%d.pt' % i)
    command += " -batch_size %d" % batchSize
    command += " -valid_batch_size %d" % batchSize
    command += " -train_steps %d" % totalTrainSteps
    command += " -report_every %d" % trainingSteps
    command += " -valid_steps %d" % trainingSteps
    command += " -optim adam"
    command += " -learning_rate_decay 0"
    command += " -log_file %s" % TEMP_LOG_PATH

    for paramName in member:
        command += " -%s %f" % (paramName, member[paramName])

    subprocess.call(command, shell=True)

    accuracy = getAccuracyFromLog()
    os.remove(TEMP_LOG_PATH)
    return accuracy


def copyModels(i, populationSize):
    src = os.path.join(modelsPath, 'genetic_%d.pt' % i)
    
    for j in range(populationSize):
        if j == i:
            continue

        dst = os.path.join(modelsPath, 'genetic_%d.pt' % j)
        copyfile(src, dst)
