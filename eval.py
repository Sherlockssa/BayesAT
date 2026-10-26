import argparse
import json
import os
import pickle
import socket
import torch
from core.data import get_data_info
from core.data import load_test_data
from core.metrics import _METRICS
from core.models import create_model
from core.trainer import Logger, seed
from core.trainer.config import args, load_config

# Setup
load_config(train=False)
attack_name = args.attack_name
if not os.path.exists(args.desc):
    print('File not found.')
    exit()

LOG_DIR = os.path.join(args.desc, args.log_dir, )
if not os.path.isdir(LOG_DIR):
    os.mkdir(os.path.join(LOG_DIR))

test_log_path = os.path.join(LOG_DIR, 'log-test-{}.log'.format(attack_name))
stats_path = os.path.join(LOG_DIR, 'eval_stats.pkl')
weights = {'state-last.pt': 'last', 'weights-best.pt': 'best'}
test_save_paths = [os.path.join(LOG_DIR, f'{attack_name}_{v}.pt') for v in weights.values()]

if os.path.exists(stats_path):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    if attack_name in stats and args.save and any(os.path.exists(p) for p in test_save_paths):
        print('Already tested.')
        exit()

for path in [test_log_path] + test_save_paths:
    if os.path.exists(path):
        os.remove(path)
logger = Logger(test_log_path)

host_info = "# " + ("%30s" % "Host Name") + ":\t" + socket.gethostname()
logger.log("#" * 120)
logger.log("----------Configurable Parameters In this Model----------")
logger.log(host_info)
for k in args.get_dict():
    logger.log("# " + ("%30s" % k) + ":\t" + str(args.__getattr__(k)))
logger.log("#" * 120)

ARG_PATH = os.path.join(args.desc, 'args.txt')
with open(ARG_PATH, 'r') as f:
    cfg_special = json.load(f)

model_train_seed = cfg_special['seed']
logger.log(f'Model training seed {model_train_seed}.')

all_keys = args.get_dict()
for k, v in cfg_special.items():
    if k not in all_keys:
        if isinstance(v, bool):
            args.DEFINE_boolean(f"-{k}", f"--{k}", default=argparse.SUPPRESS)
        else:
            args.DEFINE_argument(f"-{k}", f"--{k}", default=argparse.SUPPRESS, type=type(v))
        args.__setattr__(k, v)
        print(f"OLD ARG: {k} with value {v}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log('Using device: {}'.format(device))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
torch.backends.cudnn.benchmark = True

DATA_DIR = os.path.join(args.data_dir, args.data)
info = get_data_info(DATA_DIR)
test_dataset, test_dataloader = load_test_data(DATA_DIR, batch_size_test=args.batch_size_validation, num_workers=0)
model = create_model(args, info, device, None)

for w in weights:
    WEIGHTS = os.path.join(args.desc, w)
    if not os.path.exists(WEIGHTS):
        logger.log(f"Warning: Weight file {WEIGHTS} does not exist, skipping...")
        continue

    checkpoint = torch.load(WEIGHTS, weights_only=False, map_location=device)

    logger.log(f'Loading checkpoint from {WEIGHTS}')

    # 检查checkpoint的键，看看我们有哪些可用信息
    logger.log(f'Checkpoint keys: {list(checkpoint.keys())}')

    # 根据我们的保存格式，尝试加载s2模型权重
    # 我们可能保存了多种格式，需要逐一尝试
    state_dict = None
    epoch = None

    # 尝试1: 标准格式（评估代码期望的格式）
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 0)
        logger.log('Loaded using standard format (model_state_dict)')

    # 尝试2: 我们的完整训练状态格式
    elif 's2_model_state_dict' in checkpoint:
        state_dict = checkpoint['s2_model_state_dict']
        epoch = checkpoint.get('epoch', 0)
        logger.log('Loaded using our training format (s2_model_state_dict)')

        # 如果这是完整的训练状态，记录其他信息
        if 'current_stage' in checkpoint:
            logger.log(f'Model was saved during {checkpoint["current_stage"]} stage')

    # 尝试3: 可能是stage2特定的格式
    elif 's2_model_state_dict' not in checkpoint and 'is_stage2' in checkpoint:
        # 这可能是一个简化的stage2模型文件，需要查找其他可能的键
        for key in checkpoint.keys():
            if 'state_dict' in key and 's1' not in key and 't1' not in key:
                # 可能是model_state_dict或其他变体
                state_dict = checkpoint[key]
                epoch = checkpoint.get('epoch', 0)
                logger.log(f'Loaded using key: {key}')
                break

    if state_dict is None:
        # 如果以上都不匹配，尝试最后一个方法：查找任何包含state_dict的键
        for key in checkpoint.keys():
            if 'state_dict' in key.lower():
                state_dict = checkpoint[key]
                epoch = checkpoint.get('epoch', 0)
                logger.log(f'Loaded using fallback key: {key}')
                break

    if state_dict is None:
        logger.log(f'Error: Could not find model weights in checkpoint {WEIGHTS}')
        # 打印所有键以便调试
        logger.log(f'All keys in checkpoint: {list(checkpoint.keys())}')
        continue  # 跳过这个文件，继续下一个

    # 加载模型权重
    try:
        model.load_state_dict(state_dict)
        logger.log(f'Successfully loaded model from {WEIGHTS} (epoch-{epoch})')

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log(f'Model info: {total_params:,} total params, {trainable_params:,} trainable params')

    except Exception as e:
        logger.log(f'Error loading model state dict: {e}')
        # 尝试打印不匹配的键
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        logger.log(f'Model missing keys: {model_keys - checkpoint_keys}')
        logger.log(f'Checkpoint extra keys: {checkpoint_keys - model_keys}')
        continue



    num_classes = info['num_classes']
    logger.log(f'Begin evaluation: {weights[w]}-{attack_name}.')
    attack_opt = args.attack_opt
    if attack_name == 'aa':
        attack_opt['n_classes'] = num_classes
    adversary = _METRICS[attack_name](model, **attack_opt)

    seed(args.seed)
    model.eval()
    acc, _, _ = adversary.save(data_loader=test_dataloader, verbose=True, return_verbose=True,
                               save_path=test_save_paths[weights[w]] if args.save else None)

    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            logger.stats = pickle.load(f)
    logger.add(category=weights[w], k=attack_name, v=acc, global_it=model_train_seed, unique=True)
    logger.log('Adversarial {}-{}: {:.5f}%'.format(weights[w], attack_name, acc))
    logger.save_stats('eval_stats.pkl')
    logger.log('\nTesting completed for {} model.\n'.format(weights[w]))

# 如果没有成功加载任何模型，给出更详细的错误信息
if not os.path.exists(os.path.join(args.desc, 'state-last.pt')) and not os.path.exists(
        os.path.join(args.desc, 'weights-best.pt')):
    logger.log('=' * 80)
    logger.log('ERROR: No model files found for evaluation!')
    logger.log(f'Looking for files in: {args.desc}')
    logger.log('Expected files:')
    logger.log('  - state-last.pt')
    logger.log('  - weights-best.pt')
    logger.log('')
    logger.log('Check if training completed successfully.')
    logger.log('=' * 80)