"""
5.4.1 节对比实验
"""

from main import main as bf_main
from bert_finetune import Configs
import argparse


exp_name = "exp541"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=None)
    args = parser.parse_args()

    if args.exp == 0:
        config = Configs(from_begin=True,
                         save_model=True,
                         premodel_path="../data/model/bert-base-uncased",
                         model_file_name='{}-{}bfSdzAj.pt'.format(exp_name, args.exp),
                         batch_size=128,
                         learning_rate=5e-5,
                         shuffle=True,
                         seed=0,
                         data='../data/sdz/AJfiltered-processed-data.csv',
                         src_type='sdz_aj',
                         epochs=20,
                         device='cuda:4',
                         output_dir='./save')
    elif args.exp == 1:
        config = Configs(from_begin=True,
                         save_model=True,
                         premodel_path="../data/model/bert-base-uncased",
                         model_file_name='{}-{}bfSdzPisa.pt'.format(exp_name, args.exp),
                         batch_size=128,
                         learning_rate=5e-5,
                         shuffle=True,
                         seed=0,
                         data='../data/sdz/sdzPISAcoded.csv',
                         src_type='sdz_pisa',
                         epochs=20,
                         device='cuda:5',
                         output_dir='./save')
    elif args.exp == 2:
        config = Configs(from_begin=True,
                         save_model=True,
                         premodel_path="../data/model/bert-base-chinese",
                         model_file_name='{}-{}bfMdlaPisa.pt'.format(exp_name, args.exp),
                         batch_size=128,
                         learning_rate=5e-5,
                         shuffle=True,
                         seed=0,
                         data='../data/mdla/mdla-pisa-train.csv',
                         src_type='mdla_pisa',
                         epochs=20,
                         device='cuda:6',
                         output_dir='./save')
    else:
        return

    if args.cuda is not None:
        config.device = 'cuda:{}'.format(args.cuda)
    bf_main(config)


if __name__ == '__main__':
    main()
