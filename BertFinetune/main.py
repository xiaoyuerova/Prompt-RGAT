import os

import torch

from bert_finetune import *
from bert_finetune.utils import set_seed, get_logger, load_src_for_train, save_pkl, load_pkl


def main(args: Configs):
    set_seed(args.seed)
    log = get_logger(name=args.model_file_name[:-3])

    for attribute in dir(args):
        if not attribute.startswith('__'):  # 过滤掉内置属性
            value = getattr(args, attribute)
            log.info(f'{attribute}: {value}')

    # load data
    log.debug("Loading data from '%s'." % args.data)
    if args.from_begin:
        data_src = load_src_for_train(args.data, args, file_type='csv', shuffle=True)
        log.debug("data_src: {}".format(len(data_src)))
        train_src = data_src[:int(len(data_src) * 0.70)]
        valid_src = data_src[int(len(data_src) * 0.70):int(len(data_src) * 0.85)]
        test_src = data_src[int(len(data_src) * 0.85):]
        train_dataset = MyDataset(train_src, args.tokenizer)
        valid_dataset = MyDataset(valid_src, args.tokenizer)
        test_dataset = MyDataset(test_src, args.tokenizer)
        train_data = Data(train_dataset, args)
        dev_data = Data(valid_dataset, args)
        test_data = Data(test_dataset, args)
        save_pkl(train_data, os.path.join(args.output_dir, args.model_file_name[:-3] + '_train_data.pkl'))
        save_pkl(dev_data, os.path.join(args.output_dir, args.model_file_name[:-3] + '_valid_data.pkl'))
        save_pkl(test_data, os.path.join(args.output_dir, args.model_file_name[:-3] + '_test_data.pkl'))
    else:
        train_data = load_pkl(os.path.join(args.output_dir, args.model_file_name[:-3] + '_train_data.pkl'))
        dev_data = load_pkl(os.path.join(args.output_dir, args.model_file_name[:-3] + '_valid_data.pkl'))
        test_data = load_pkl(os.path.join(args.output_dir, args.model_file_name[:-3] + '_test_data.pkl'))

    log.debug("Building model...")
    model_file = os.path.join(args.output_dir, args.model_file_name)
    model = BertFinetune(args).to(args.device)
    opt = Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)

    coach = Coach(train_data, dev_data, test_data, model, opt, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)

    # Train.
    log.info("Start training...")
    ret = coach.train()

    # Save.
    if args.save_model:
        log.info("Saving model to %s..." % model_file)
        checkpoint = {
            "best_dev_f1": ret[0],
            "best_epoch": ret[1],
            "best_state": ret[2],
        }
        torch.save(checkpoint, model_file)


if __name__ == '__main__':
    config = Configs(from_begin=True,
                     save_model=True,
                     premodel_path="../data/model/bert-base-uncased",
                     model_file_name='bfSdzAjTest.pt',
                     batch_size=128,
                     learning_rate=5e-5,
                     max_length=64,
                     shuffle=True,
                     seed=0,
                     data='../data/sdz/AJfiltered-processed-data.csv',
                     src_type='sdz_aj',
                     epochs=20,
                     device='cuda:0',
                     output_dir='./save')
    main(config)
