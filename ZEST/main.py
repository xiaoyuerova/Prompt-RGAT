import os

import torch

from SPCL import Configs, MyDataset, Coach, SPCL, Optim, load_src_for_train
from SPCL.utils import get_params_group, set_seed, save_pkl, load_pkl, get_logger


def main(args: Configs):
    set_seed(args.seed)
    logger = get_logger(args.model_file_name[:-3], level='INFO')

    for attribute in dir(args):
        if not attribute.startswith('__'):  # 过滤掉内置属性
            value = getattr(args, attribute)
            logger.info(f'{attribute}: {value}')

    logger.debug("Loading data from '%s'." % args.data)
    if args.from_begin:
        data_src = load_src_for_train(args.data, args, file_type='csv', shuffle=True)
        logger.debug("data_src: {}".format(len(data_src)))
        train_src = data_src[:int(len(data_src) * 0.70)]
        valid_src = data_src[int(len(data_src) * 0.70):int(len(data_src) * 0.85)]
        test_src = data_src[int(len(data_src) * 0.85):]
        save_pkl(train_src, os.path.join(args.output_dir, args.model_file_name + '_train_src.pkl'))
        save_pkl(valid_src, os.path.join(args.output_dir, args.model_file_name + '_valid_src.pkl'))
        save_pkl(test_src, os.path.join(args.output_dir, args.model_file_name + '_test_src.pkl'))
    else:
        train_src = load_pkl(os.path.join(args.output_dir, args.model_file_name + '_train_src.pkl'))
        valid_src = load_pkl(os.path.join(args.output_dir, args.model_file_name + '_valid_src.pkl'))
        test_src = load_pkl(os.path.join(args.output_dir, args.model_file_name + '_test_src.pkl'))

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

    if args.save_model:
        logger.info("Saving model to %s..." % model_file)
        checkpoint = {
            "best_direct_f1": ret[0],
            "best_cluster_f1": ret[1],
            "best_epoch": ret[2],
            "best_state": ret[3],
        }
        torch.save(checkpoint, model_file)


if __name__ == '__main__':
    # TRAIN_OBJ = [0]*12 + [1, 0]*4 + [0, 0]
    # TRAIN_OBJ = [0] * 20
    config = Configs(from_begin=True,
                     save_model=True,
                     model_file_name='zseqSazAjTest.pt',
                     premodel_path='../data/model/bert-base-uncased',
                     batch_size=64,
                     learning_rate=1e-4,
                     seed=0,
                     data='../data/sdz/AJfiltered-processed-data.csv',
                     src_type='sdz_aj',
                     epochs=10,
                     train_obj='ce',   # 'ce', 'spcl', 'spdcl'
                     cl_selection_ratio=1,
                     cl=True,
                     max_utterance_history_len=6,
                     extra_target_ratio=0.5,
                     device='cuda:7',
                     output_dir='./save', )
    main(config)

    # for i in range(4):
    #     if i % 2 == 0:
    #         config.from_begin = False
    #         config.epochs = 3
    #         config.train_obj = 'spcl'
    #         main(config)
    #     else:
    #         config.from_begin = False
    #         config.epochs = 3
    #         config.train_obj = 'ce'
    #         main(config)
    #
    # config.from_begin = False
    # config.epochs = 5
    # config.train_obj = 'ce'
    # main(config)
