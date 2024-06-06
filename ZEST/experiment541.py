from main import main as spcl_train
from SPCL import Configs
import argparse

exp_name = "exp541"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=6)
    parser.add_argument('--cuda', type=int, default=None)
    args = parser.parse_args()

    if args.exp == 6:
        config = Configs(from_begin=True,
                         save_model=True,
                         model_file_name='{}-{}zseqSdzAj.pt'.format(exp_name, args.exp),
                         premodel_path='../data/model/bert-base-uncased',
                         batch_size=64,
                         learning_rate=5e-5,
                         seed=0,
                         data='../data/sdz/AJfiltered-processed-data.csv',
                         src_type='sdz_aj',
                         epochs=len(TRAIN_OBJ),
                         train_obj=TRAIN_OBJ,  # 'ce', 'spcl', 'spdcl'
                         shuffle=False,
                         max_utterance_history_len=6,
                         extra_target_ratio=0.5,
                         device='cuda:4',
                         output_dir='./save', )
    elif args.exp == 7:
        config = Configs(from_begin=True,
                         save_model=True,
                         model_file_name='{}-{}zseqSdzPisa.pt'.format(exp_name, args.exp),
                         premodel_path='../data/model/bert-base-uncased',
                         batch_size=64,
                         learning_rate=5e-5,
                         seed=0,
                         data='../data/sdz/sdzPISAcoded.csv',
                         src_type='sdz_pisa',
                         epochs=len(TRAIN_OBJ),
                         train_obj=TRAIN_OBJ,  # 'ce', 'spcl', 'spdcl'
                         shuffle=False,
                         max_utterance_history_len=6,
                         extra_target_ratio=0.5,
                         device='cuda:5',
                         output_dir='./save', )
    else:
        config = Configs(from_begin=True,
                         save_model=True,
                         model_file_name='{}-{}zseqMdlaPisa.pt'.format(exp_name, args.exp),
                         premodel_path='../data/model/bert-base-chinese',
                         batch_size=64,
                         learning_rate=5e-5,
                         seed=0,
                         data='../data/mdla/mdla-pisa-train.csv',
                         src_type='mdla_pisa',
                         epochs=len(TRAIN_OBJ),
                         train_obj=TRAIN_OBJ,  # 'ce', 'spcl', 'spdcl'
                         shuffle=False,
                         max_utterance_history_len=6,
                         extra_target_ratio=0.5,
                         device='cuda:6',
                         output_dir='./save', )

    if args.cuda is not None:
        config.device = 'cuda:{}'.format(args.cuda)
        config.device = 'cuda:{}'.format(args.cuda)
        config.device = 'cuda:{}'.format(args.cuda)
    spcl_train(config)


if __name__ == "__main__":
    main()
