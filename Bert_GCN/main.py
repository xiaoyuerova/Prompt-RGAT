import os
import torch
import torch.utils.data as data
from BertGCN import *
from BertGCN.utils import get_logger, set_seed, save_pkl, load_pkl, DataType


def main(args: Parameter):
    set_seed(args.seed)
    log = get_logger(args.model_file_name[:-3], level='INFO')

    # load data
    log.debug("Loading data from '%s'." % args.data)
    dataset = MyDataset(load_pkl(args.data))
    log.info("Loaded data.")

    total = dataset.__len__()
    lengths = [int(0.7 * total), int(0.15 * total)]
    lengths.append(total - lengths[0] - lengths[1])

    train_dataset, valid_dataset, test_dataset = data.random_split(
        dataset=dataset,
        lengths=lengths,
        generator=torch.Generator().manual_seed(args.seed)
    )
    print('数据划分验证', train_dataset[0].text_src[0])
    print('数据划分验证', valid_dataset[0].text_src[0])
    print('数据划分验证', test_dataset[0].text_src[0])

    train_data = Data(train_dataset, args)
    dev_data = Data(valid_dataset, args)
    test_data = Data(test_dataset, args)

    log.debug("Building model...")
    model_file = os.path.join(args.output_dir, args.model_file_name)
    model = BertGCN(args.text_dim, args).to(args.device)
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
        log.info("Saving model...")
        checkpoint = {
            "best_dev_f1": ret[0],
            "best_epoch": ret[1],
            "best_state": ret[2],
        }
        torch.save(checkpoint, model_file)


if __name__ == "__main__":
    # parameters = Parameter(from_begin=True,
    #                        model_file_name='{}-{}gcnZseqSdzAj.pt'.format('exp542', 'test'),
    #                        save_model=True,
    #                        batch_size=4,
    #                        learning_rate=1e-4,
    #                        src_type='sdz_aj',
    #                        seed=0,
    #                        epochs=20,
    #                        data='../ZEST/save/exp543-0zseqSdzAj_features.pkl',
    #                        device='cuda:7',
    #                        wp=8,
    #                        wf=4,
    #                        output_dir='./save')

    parameters = Parameter(from_begin=True,
                           model_file_name='{}-{}gcnZseqSdzPisa.pt'.format('exp542', 'test'),
                           save_model=True,
                           batch_size=4,
                           learning_rate=1e-4,
                           src_type='sdz_pisa',
                           seed=0,
                           epochs=20,
                           data='../ZEST/save/exp543-3zseqSdzPisa_features.pkl',
                           device='cuda:7',
                           wp=8,
                           wf=4,
                           output_dir='./save')
    main(parameters)
