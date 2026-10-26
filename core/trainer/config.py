import argparse
import re

import yaml

from core.trainer.utils import str2float
from tunner import ARGS_FOR_TUNE
from .args import _ArgValues, _global_parser
from .parser import set_parser_train, set_parser_eval

'''
Argument Definition:
config_file : file to config argument
'''
args = _ArgValues()
args.DEFINE_argument("config_file", type=str, help="Path to config file.",)

def load_config(train=True):
    (set_parser_train if train else set_parser_eval)(_global_parser)

    with open(args.config_file, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    input_arguments, all_keys = [], args.get_dict()

    for k, v in cfg_special.items():
        if k in all_keys:
            if v == all_keys[k]:
                continue
            if k in ARGS_FOR_TUNE:
                print(f"Note!: tuning args: {k} with value {getattr(args, k)}")
                continue
            setattr(args, k, v)
            input_arguments.append(k)
        else:
            define_fn = args.DEFINE_boolean if isinstance(v, bool) else args.DEFINE_argument
            kwargs = {"default": argparse.SUPPRESS}
            if not isinstance(v, bool):
                kwargs["type"] = type(v)
            define_fn(f"-{k}", f"--{k}", **kwargs)
            setattr(args, k, v)
            print(f"Note!: new args: {k} with value {getattr(args, k)}")

    check(args, train)
    return input_arguments

def check(args, train):
    rule_list = [path_rule, atk_para_rule] if train else [atk_para_rule_test]
    def recursive_check(k, v, rule_list):
        if isinstance(v, dict):
            return {sub_k: recursive_check(sub_k, sub_v, rule_list) for sub_k, sub_v in v.items()}
        if isinstance(v, list):
            return [recursive_check(k, v_i, rule_list) for v_i in v]
        for rule in rule_list:
            v = rule(k, v)
        return v

    for k, v in args.get_dict().items():
        setattr(args, k, recursive_check(k, v, rule_list))

def path_rule(k, v):
    if any(x in k for x in ["path", "desc", "dir"]):
        mapping = {
            "DATAINFO": f"{args.data}",
            "MODELINFO": f"{args.model}",
            "METHODINFO": f"{args.method}",
            "SEEDINFO": f"seed{args.seed}",
        }
        for token, repl in mapping.items():
            v = re.sub(token, repl, v)
    return v

def atk_para_rule(k, v):
    return str2float(v) if k in {"attack_eps", "attack_step"} else v

def atk_para_rule_test(k, v):
    return str2float(v) if k in {"eps", "alpha"} else v