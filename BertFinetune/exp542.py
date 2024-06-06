from .gen_feature import main as gen_feature_main
from .gen_feature import init_by_dataset
from bert_finetune import Configs
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=None)
    args = parser.parse_args()

    sdz_config = Configs(premodel_path="../data/model/bert-base-uncased",
                         model_file_name='exp541-0bfSdzAj.pt',
                         src_type='sdz_aj',
                         batch_size=128,
                         seed=0,
                         device='cuda:4',
                         output_dir='./save')

    sdzp_config = Configs(premodel_path="../data/model/bert-base-uncased",
                          model_file_name='exp541-1bfSdzPisa.pt',
                          src_type='sdz_pisa',
                          batch_size=128,
                          seed=0,
                          device='cuda:5',
                          output_dir='./save')

    mdla_config = Configs(premodel_path='../data/model/bert-base-chinese',
                          model_file_name='exp541-2bfMdlaPisa.pt',
                          src_type='mdla_pisa',
                          batch_size=128,
                          seed=0,
                          device='cuda:6',
                          output_dir='./save')

    if args.exp == 1:
        config = sdz_config
    elif args.exp == 3:
        config = sdzp_config
    elif args.exp == 5:
        config = mdla_config
    else:
        return

    config = init_by_dataset(args=config)
    if args.cuda is not None:
        config.device = 'cuda:{}'.format(args.cuda)
    gen_feature_main(config)


if __name__ == '__main__':
    main()
