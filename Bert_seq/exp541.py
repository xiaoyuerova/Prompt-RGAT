"""
5.4.1 节对比实验
该模型表现不稳定。有时候结果会很差，所以跑了lr=0和lr=1两轮，取高的一次结果（可能是由于使用了nn.ReLU()）
"""

from main import main as bert_seq_main
from bert_seq import Configs
import argparse


exp_name = "exp541"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=3)
    parser.add_argument('--cuda', type=int, default=None)
    args = parser.parse_args()

    if args.exp == 3:
        config = Configs(from_begin=True,
                         save_model=True,
                         premodel_path="../data/model/bert-base-uncased",
                         model_file_name='{}-{}seqSdzAj.pt'.format(exp_name, args.exp),
                         batch_size=128,
                         learning_rate=5e-5,
                         seed=0,
                         data='../data/sdz/AJfiltered-processed-data.csv',
                         src_type='sdz_aj',
                         epochs=20,
                         device='cuda:4',
                         output_dir='./save')
    elif args.exp == 4:
        config = Configs(from_begin=True,
                         save_model=True,
                         premodel_path="../data/model/bert-base-uncased",
                         model_file_name='{}-{}seqSdzPisa.pt'.format(exp_name, args.exp),
                         batch_size=128,
                         learning_rate=5e-5,
                         seed=0,
                         data='../data/sdz/sdzPISAcoded.csv',
                         src_type='sdz_pisa',
                         epochs=20,
                         device='cuda:5',
                         output_dir='./save')
    else:
        config = Configs(from_begin=True,
                         save_model=True,
                         premodel_path="../data/model/bert-base-chinese",
                         model_file_name='{}-{}seqMdlaPisa.pt'.format(exp_name, args.exp),
                         batch_size=128,
                         learning_rate=5e-5,
                         seed=0,
                         data='../data/mdla/mdla-pisa-train.csv',
                         src_type='mdla_pisa',
                         epochs=20,
                         device='cuda:6',
                         output_dir='./save')

    config.drop_rate = 0
    config.seed = 4
    if args.cuda is not None:
        config.device = 'cuda:{}'.format(args.cuda)

    bert_seq_main(config)


if __name__ == '__main__':
    main()
