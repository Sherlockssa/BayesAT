import itertools
import os

methods = ['mt']
datasets = ['cifar10']
ARGS_FOR_TUNE = dict(
    seed = [1],
    model = ['wrn-28-10',],
    # adv_eval_freq = [1],
)

if __name__ == '__main__':
    gpu = 4
    commands = []
    groups = []
    for method, dataset in itertools.product(methods, datasets):
        groups.append(f"python -X faulthandler train_mt.py ./configs/{method}_{dataset}.yaml")

    command_template = " {}"
    for k in ARGS_FOR_TUNE:
        command_template += " --" + k + " {}"
    possible_value = [groups]
    for k in ARGS_FOR_TUNE:
        possible_value.append(ARGS_FOR_TUNE[k])
    for args in itertools.product(*possible_value):
        commands.append(command_template.format(*args))
    print(commands)
    print("# experiments = {}".format(len(commands)))
    def exp_runner(com, gpu):
        return os.system("CUDA_VISIBLE_DEVICES={} ".format(gpu) + com)
    for com in commands:
        exp_runner(com, gpu)

