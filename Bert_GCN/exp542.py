"""
exp     dataset      Model
        Sdz_AJ
0                    ZEST（Context Prompt＋RGAT）
1                    ZEST（RGAT）
                     ZEST（Context Prompt）
                     Prompt-Based Finetune
        Sdz_PISA
2                    ZEST（Context Prompt＋RGAT）
3                    ZEST（RGAT）
                     ZEST（Context Prompt）
                     Prompt-Based Finetune
        Mdla_PISA
4                    ZEST（Context Prompt＋RGAT）
5                    ZEST（RGAT）
                     ZEST（Context Prompt）
                     Prompt-Based Finetune
"""

import argparse
from main import main as bgcn_main
from BertGCN import Parameter
from BertGCN.utils import DataType

exp_name = 'exp542'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=None)
    args = parser.parse_args()

    if args.exp == 0:
        parameters = Parameter(from_begin=True,
                               model_file_name='{}-{}gcnZseqSdzAj.pt'.format(exp_name, args.exp),
                               save_model=True,
                               batch_size=4,
                               learning_rate=1e-4,
                               src_type='sdz_aj',
                               seed=0,
                               epochs=20,
                               data='../ZEST/save/exp543-0zseqSdzAj_features.pkl',
                               device='cuda:4',
                               wp=8,
                               wf=4,
                               output_dir='./save')
    elif args.exp == 2:
        parameters = Parameter(from_begin=True,
                               model_file_name='{}-{}gcnZseqSdzPisa.pt'.format(exp_name, args.exp),
                               save_model=True,
                               batch_size=4,
                               learning_rate=1e-4,
                               src_type='sdz_pisa',
                               seed=0,
                               epochs=20,
                               data='../ZEST/save/exp543-3zseqSdzPisa_features.pkl',
                               device='cuda:5',
                               wp=8,
                               wf=4,
                               output_dir='./save')

    elif args.exp == 4:
        parameters = Parameter(from_begin=True,
                               model_file_name='{}-{}gcnZseqMdlaPisa.pt'.format(exp_name, args.exp),
                               save_model=True,
                               batch_size=4,
                               learning_rate=1e-4,
                               src_type='mdla_pisa',
                               seed=0,
                               epochs=20,
                               data='../ZEST/save/exp543-6zseqMdlaPisa_features.pkl',
                               device='cuda:6',
                               wp=8,
                               wf=4,
                               output_dir='./save')
    elif args.exp == 1:
        # ./save/exp541-0bfSdzAj_features.pkl
        parameters = Parameter(from_begin=True,
                               model_file_name='{}-{}gcnBfSdzAj.pt'.format(exp_name, args.exp),
                               save_model=True,
                               batch_size=4,
                               learning_rate=1e-4,
                               src_type='sdz_aj',
                               seed=0,
                               epochs=20,
                               data='../BertFinetune/save/exp541-0bfSdzAj_features.pkl',
                               device='cuda:4',
                               wp=8,
                               wf=4,
                               output_dir='./save')
    elif args.exp == 3:
        parameters = Parameter(from_begin=True,
                               model_file_name='{}-{}gcnBfSdzPisa.pt'.format(exp_name, args.exp),
                               save_model=True,
                               batch_size=4,
                               learning_rate=1e-4,
                               src_type='sdz_pisa',
                               seed=0,
                               epochs=20,
                               data='../BertFinetune/save/exp541-1bfSdzPisa_features.pkl',
                               device='cuda:5',
                               wp=8,
                               wf=4,
                               output_dir='./save')

    elif args.exp == 5:
        parameters = Parameter(from_begin=True,
                               model_file_name='{}-{}gcnBfMdlaPisa.pt'.format(exp_name, args.exp),
                               save_model=True,
                               batch_size=4,
                               learning_rate=1e-4,
                               src_type='mdla_pisa',
                               seed=0,
                               epochs=20,
                               data='../BertFinetune/save/exp541-2bfMdlaPisa_features.pkl',
                               device='cuda:6',
                               wp=8,
                               wf=4,
                               output_dir='./save')
    else:
        return
    parameters.drop_rate = 0
    parameters.seed = 0
    bgcn_main(parameters)


if __name__ == '__main__':
    main()
