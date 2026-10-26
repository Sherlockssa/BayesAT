import itertools
import os

model_paths = []
model_path_temp = 'trained_models/{}/{}_{}_seed{}'
ARGS = dict(
    dataset = ['cifar10'],
    method = ['BayesAT'],
    model = ['wrn-28-10'],
    seed = [1],
    # version=['P'],
)
possible_value = []
for k in ARGS:
    possible_value.append(ARGS[k])
for args in itertools.product(*possible_value):
    model_paths.append(model_path_temp.format(*args))


eval_attacks = ['nat', 'fgsm', 'pgd', 'cw', 'aa']
EVAL_ARGS_FOR_TUNE = {
    "desc": model_paths,
    'batch-size-validation': [128],
}
if __name__ == '__main__':
    gpu = 5
    commands = []
    tunner_groups = []
    for atk in eval_attacks:
        tunner_groups.append(f"python eval.py ./configs/eval/{atk}.yaml", )

    command_template = " {}"
    for k in EVAL_ARGS_FOR_TUNE:
        command_template += " --" + k + " {}"
    possible_value = []
    possible_value.append(tunner_groups)
    for k in EVAL_ARGS_FOR_TUNE:
        possible_value.append(EVAL_ARGS_FOR_TUNE[k])
    for args in itertools.product(*possible_value):
        commands.append(command_template.format(*args))
    print(commands)
    print("# experiments = {}".format(len(commands)))

    def exp_runner(com, gpu):
        return os.system("CUDA_VISIBLE_DEVICES={} ".format(gpu) + com)
    for com in commands:
        exp_runner(com, gpu)

