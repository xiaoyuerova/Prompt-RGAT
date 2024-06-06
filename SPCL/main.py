import os

import torch

from SPCL import Configs, MyDataset, Coach, SPCL, Optim, load_src
from SPCL.utils import get_params_group
from utils import set_seed, save_pkl, load_pkl, get_logger

logger = get_logger('SPCL', level='INFO')

def main(args: Configs):
    set_seed(args.seed)
    for attribute in dir(args):
        if not attribute.startswith('__'):  # 过滤掉内置属性
            value = getattr(args, attribute)
            logger.info(f'{attribute}: {value}')

    logger.debug("Loading data from '%s'." % args.data)
    if args.from_begin:
        data_src = load_src(args.data, args, shuffle=True, replace_name=False)
        logger.debug("data_src: {}".format(len(data_src)))
        train_src = data_src[:int(len(data_src) * 0.70)]
        valid_src = data_src[int(len(data_src) * 0.70):int(len(data_src) * 0.85)]
        test_src = data_src[int(len(data_src) * 0.85):]
        save_pkl(train_src, './save/train_src.pkl')
        save_pkl(valid_src, './save/valid_src.pkl')
        save_pkl(test_src, './save/test_src.pkl')
    else:
        train_src = load_pkl('./save/train_src.pkl')
        valid_src = load_pkl('./save/valid_src.pkl')
        test_src = load_pkl('./save/test_src.pkl')

    train_dataset = MyDataset(train_src, args, train=True)
    valid_dataset = MyDataset(valid_src, args, train=False)
    test_dataset = MyDataset(test_src, args, train=False)

    logger.debug("Train data: %d, Valid data: %d, Test data: %d" % (len(
        train_dataset), len(valid_dataset), len(test_dataset)))

    logger.debug("Building model...")
    model_file = os.path.join(args.output_dir, args.model_file_name)
    model = SPCL(args).to(args.device)

    optimizer = torch.optim.AdamW(get_params_group(model, args))

    coach = Coach(train_dataset, valid_dataset, test_dataset, model, optimizer, args)

    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)

    # Train.
    logger.info("Start training...")
    ret = coach.train()

    # Save.
    if args.save_model:
        logger.info("Saving model to %s..." % model_file)
        checkpoint = {
            "best_dev_f1": ret[0],
            "best_epoch": ret[1],
            "best_state": ret[2],
        }
        torch.save(checkpoint, model_file)


if __name__ == '__main__':
    config = Configs(from_begin=True,
                     save_model=True,
                     model_file_name='checkpoint.pt',
                     batch_size=64,
                     learning_rate=1e-4,
                     seed=1,
                     data='../data/PISA coded.csv',
                     epochs=1,
                     train_obj='spcl',    # 'ce', 'spcl', 'spdcl'
                     shuffle=False,
                     max_utterance_history_len=2,
                     extra_target_ratio=0.1,
                     device='cuda:0',
                     output_dir='./save',)

    main(config)

    # for i in range(8):
    #     if i % 2 == 1:
    #         config.from_begin = False
    #         config.epochs = 2
    #         config.train_obj = 'ce'
    #         main(config)
    #     else:
    #         config.from_begin = False
    #         config.epochs = 2
    #         config.train_obj = 'spcl'
    #         main(config)
