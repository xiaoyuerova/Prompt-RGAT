"""
exp543：cl和spcl的影响
各种方案下的参数设置:
使用CE：config.cl = True. 并设置ce策略
使用SPCL: TRAIN_OBJ = 'spcl'
(1): 设置spcl策略，不设置cl策略
交叉训练spcl和ce
config.cl = False
(2): 不设置spcl策略，设置cl策略
全部训练'ce'
config.cl = True
(3): 设置spcl策略，设置cl策略
交叉训练spcl和cl
config.cl = True
"""

from main import main as spcl_train
from SPCL import Configs
import argparse

exp_name = "exp543"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=None)
    args = parser.parse_args()

    sdz_config = Configs(from_begin=True,
                         save_model=True,
                         model_file_name='{}-{}zseqSdzAj.pt'.format(exp_name, args.exp),
                         premodel_path='../data/model/bert-base-uncased',
                         batch_size=64,
                         learning_rate=5e-5,
                         seed=0,
                         data='../data/sdz/AJfiltered-processed-data.csv',
                         src_type='sdz_aj',
                         epochs=20,
                         train_obj='ce',  # 'ce', 'spcl', 'spdcl'
                         cl=False,
                         max_utterance_history_len=6,
                         extra_target_ratio=0.5,
                         device='cuda:4',
                         output_dir='./save', )

    sdzp_config = Configs(from_begin=True,
                          save_model=True,
                          model_file_name='{}-{}zseqSdzPisa.pt'.format(exp_name, args.exp),
                          premodel_path='../data/model/bert-base-uncased',
                          batch_size=64,
                          learning_rate=5e-5,
                          seed=0,
                          data='../data/sdz/sdzPISAcoded.csv',
                          src_type='sdz_pisa',
                          epochs=20,
                          train_obj='ce',  # 'ce', 'spcl', 'spdcl'
                          cl=True,
                          max_utterance_history_len=6,
                          extra_target_ratio=0.5,
                          device='cuda:5',
                          output_dir='./save', )

    mdla_config = Configs(from_begin=True,
                          save_model=True,
                          model_file_name='{}-{}zseqMdlaPisa.pt'.format(exp_name, args.exp),
                          premodel_path='../data/model/bert-base-chinese',
                          batch_size=64,
                          learning_rate=5e-5,
                          seed=0,
                          data='../data/mdla/mdla-pisa-train.csv',
                          src_type='mdla_pisa',
                          epochs=20,
                          train_obj='ce',  # 'ce', 'spcl', 'spdcl'
                          cl=True,
                          max_utterance_history_len=6,
                          extra_target_ratio=0.5,
                          device='cuda:6',
                          output_dir='./save', )

    if args.cuda is not None:
        sdz_config.device = 'cuda:{}'.format(args.cuda)
        sdzp_config.device = 'cuda:{}'.format(args.cuda)
        mdla_config.device = 'cuda:{}'.format(args.cuda)

    """
        Sdz_AJ Sdz_PISA Mdla_PISA
    0   0       3       6
    1   7       1       4
    2   5       8       2
    3   9       10      11
    """

    if args.exp == 0:
        exp_strategy(sdz_config, strategy=0)
    elif args.exp == 1:
        exp_strategy(sdzp_config, strategy=1)
    elif args.exp == 2:
        exp_strategy(mdla_config, strategy=2)

    elif args.exp == 3:
        exp_strategy(sdzp_config, strategy=0)
    elif args.exp == 4:
        exp_strategy(mdla_config, strategy=1)
    elif args.exp == 5:
        exp_strategy(sdz_config, strategy=2)

    elif args.exp == 6:
        exp_strategy(mdla_config, strategy=0)
    elif args.exp == 7:
        exp_strategy(sdz_config, strategy=1)
    elif args.exp == 8:
        exp_strategy(sdzp_config, strategy=2)

    elif args.exp == 9:
        exp_strategy(sdz_config, strategy=3)
    elif args.exp == 10:
        exp_strategy(sdzp_config, strategy=3)
    elif args.exp == 11:
        exp_strategy(mdla_config, strategy=3)
    else:
        raise ValueError('Invalid exp')


def exp_strategy(config: Configs, strategy: int = 0):
    if strategy == 0:
        config.from_begin = True
        config.epochs = 8
        config.train_obj = 'ce'
        config.cl = False
        spcl_train(config)

        config.from_begin = False
        config.epochs = 4
        config.train_obj = 'spcl'
        spcl_train(config)

        for i in range(4):
            if i % 2 == 0:
                config.from_begin = False
                config.epochs = 4
                config.train_obj = 'ce'
                spcl_train(config)
            else:
                config.from_begin = False
                config.epochs = 4
                config.train_obj = 'spcl'
                spcl_train(config)

    elif strategy == 1:
        config.from_begin = True
        config.epochs = 30
        config.train_obj = 'ce'
        config.cl = True
        spcl_train(config)

    elif strategy == 2:
        config.from_begin = True
        config.epochs = 8
        config.train_obj = 'ce'
        config.cl = True
        spcl_train(config)

        config.from_begin = False
        config.epochs = 4
        config.train_obj = 'spcl'
        spcl_train(config)

        for i in range(4):
            if i % 2 == 0:
                config.from_begin = False
                config.epochs = 4
                config.train_obj = 'ce'
                spcl_train(config)
            else:
                config.from_begin = False
                config.epochs = 4
                config.train_obj = 'spcl'
                spcl_train(config)

    elif strategy == 3:
        config.from_begin = True
        config.epochs = 30
        config.train_obj = 'ce'
        config.cl = False
        spcl_train(config)


if __name__ == "__main__":
    main()
