import os
import torch
from tqdm import tqdm
from bert_finetune import Configs, MyDataset, Data, BertFinetune
from bert_finetune.utils import gen_all_reps, set_seed, save_pkl, load_pkl, get_logger, load_src_for_train


class DataType(object):
    def __init__(self,
                 text_src: list[str],
                 text_tensor: torch.Tensor,
                 speaker: list,
                 label: list):
        self.text_src = text_src
        self.text_tensor = text_tensor
        self.speaker = speaker
        self.label = label


def main(args: Configs):
    model = BertFinetune(args).to(args.device)
    # print(os.path.join(args.output_dir, args.model_file_name))
    ckpt = torch.load(os.path.join(args.output_dir, args.model_file_name))
    model.load_state_dict(ckpt["best_state"])
    features = []

    data_src = load_src_for_train(args.data, args, shuffle=True)
    for seq in data_src:
        dataset = MyDataset([seq], args.tokenizer)
        data = Data(dataset, args, shuffle=False)
        item = {
            'text_src': [],
            'text_tensor': None,
            'speaker': [],
            'label': []
        }
        idxes, all_reps, all_corr_labels = gen_all_reps(model, data)

        for idx in tqdm(idxes, total=len(idxes)):
            item['text_src'].append(data.dataset[idx]["texts"])
            item['speaker'].append(0)
            item['label'].append(data.dataset[idx]["labels"])
        item['text_tensor'] = all_reps.cpu()
        features.append(DataType(**item))
    # Save.
    # {'text_src': 'hi',
    #  'text': tensor([768]),
    #  'speaker': 0,
    #  'label': 12}

    save_file = os.path.join(args.output_dir, args.model_file_name[:-3] + '_features.pkl')
    save_pkl(features, save_file)
    print("Done! save to {}".format(save_file))


def init_by_dataset(args: Configs):
    src_type = args.src_type
    if src_type == 'sdz_aj':
        args.data = '../data/sdz/AJfiltered-processed-data.csv'
    elif src_type == 'sdz_pisa':
        args.data = '../data/sdz/sdzPISAcoded.csv'
    elif src_type == 'mdla_pisa':
        args.data = '../data/mdla/mdla-pisa-train.csv'
    else:
        raise NotImplementedError
    return args


exp_name = "exp541"

if __name__ == '__main__':
    config = Configs(premodel_path='../data/model/bert-base-chinese',
                     model_file_name='exp541-2bfMdlaPisa.pt',
                     src_type='mdla_pisa',
                     batch_size=128,
                     seed=0,
                     device='cuda:7',
                     output_dir='./save')

    config = init_by_dataset(config)
    main(config)
