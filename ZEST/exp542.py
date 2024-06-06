from gen_feature import main as gen_feature_main
from gen_feature import init_by_dataset
from SPCL import Configs
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=None)
    args = parser.parse_args()

    sdz_config = Configs(
        model_file_name='exp543-0zseqSdzAj.pt',
        premodel_path='../data/model/bert-base-uncased',
        src_type='sdz_aj',
        batch_size=64,
        seed=0,
        max_utterance_history_len=6,
        extra_target_ratio=0.5,
        device='cuda:4',
        output_dir='./save', )

    sdzp_config = Configs(
        model_file_name='exp543-3zseqSdzPisa.pt',
        premodel_path='../data/model/bert-base-uncased',
        src_type='sdz_pisa',
        batch_size=64,
        seed=0,
        max_utterance_history_len=6,
        extra_target_ratio=0.5,
        device='cuda:5',
        output_dir='./save', )

    mdla_config = Configs(
        model_file_name='exp543-6zseqMdlaPisa.pt',
        premodel_path='../data/model/bert-base-chinese',
        src_type='mdla_pisa',
        batch_size=64,
        seed=0,
        max_utterance_history_len=6,
        extra_target_ratio=0.5,
        device='cuda:6',
        output_dir='./save', )

    if args.exp == 0:
        config = sdz_config
    elif args.exp == 2:
        config = sdzp_config
    elif args.exp == 4:
        config = mdla_config
    else:
        return

    config = init_by_dataset(args=config)
    if args.cuda is not None:
        config.device = 'cuda:{}'.format(args.cuda)
    gen_feature_main(config)


if __name__ == '__main__':
    main()
