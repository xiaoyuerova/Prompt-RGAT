import os

import torch

from SPCL import Configs, MyDataset, Data, SPCL, Optim, load_src
from SPCL.utils import get_params_group
from utils import set_seed, save_pkl, load_pkl, get_logger
from SPCL.utils import gen_all_reps

save_path = './save/features.pkl'
model_file = './save/checkpoint.pt'

def main(args: Configs):
    data_src = load_src(args.data, args, shuffle=True, replace_name=False)
    dataset = MyDataset(data_src, args, train=False)
    data = Data(dataset, args, train=False, shuffle=False)


    model = SPCL(args).to(args.device)
    ckpt = torch.load(model_file)
    model.load_state_dict(ckpt["best_state"])

    all_reps, all_corr_labels = gen_all_reps(model, data)

    # Save.
    features = {
        "": all_reps,
        "best_epoch": ret[1],
    }
    torch.save(checkpoint, model_file)


if __name__ == '__main__':
    config = Configs(from_begin=True,
                     save_model=True,
                     model_file_name='checkpoint.pt',
                     batch_size=64,
                     learning_rate=1e-4,
                     seed=1,
                     data='../data/PISA coded.csv',
                     epochs=10,
                     train_obj='spcl',    # 'ce', 'spcl', 'spdcl'
                     shuffle=False,
                     max_utterance_history_len=6,
                     extra_target_ratio=0.5,
                     device='cuda:0',
                     output_dir='./save',)

    main(config)

    # for i in range(8):
    #     if i % 2 == 1:
    #         config.from_begin = False
    #         config.epochs = 2
    #         config.train_obj = 'ce'
    #         main(config)
    #     else:
    #         config.from_begin = False
    #         config.epochs = 2
    #         config.train_obj = 'spcl'
    #         main(config)
