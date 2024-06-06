import torch
from SPCL import *
from utils import load_pkl


def main():
    model_file = './save/ZEST_zh.pt'
    args = Configs(from_begin=True,
                   save_model=True,
                   model_file_name='ZEST_zh.pt',
                   premodel_path='bert-base-chinese',
                   batch_size=64,
                   learning_rate=1e-4,
                   seed=1,
                   data='../data/mdla/mdla-pisa-train.csv',
                   src_type='mdla_zh',
                   epochs=15,
                   train_obj='ce',  # 'ce', 'spcl', 'spdcl'
                   shuffle=False,
                   max_utterance_history_len=6,
                   extra_target_ratio=0.5,
                   device='cuda:0',
                   output_dir='./save', )
    model = SPCL(args).to(args.device)

    ckpt = torch.load(model_file)
    # best_dev_f1 = ckpt["best_dev_f1"]
    # best_epoch = ckpt["best_epoch"]
    best_state = ckpt["best_state"]
    model.load_state_dict(best_state)
    # test_src = load_pkl("./save/test_src.pkl")
    # test_dataset = MyDataset(test_src, args, train=False)
    # test_data = Data(test_dataset, args, train=False, shuffle=False)
    # print(model.centers_mask)
    # for idx, batch in enumerate(test_data.dataloader):
    #     out = model.forward(batch)
    #     print(out)
    #     break


if __name__ == '__main__':
    main()
