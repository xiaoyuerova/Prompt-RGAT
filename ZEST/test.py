from main import main as spcl_train
from SPCL import Configs


def main():
    TRAIN_OBJ = [0] * 6 + [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0] + [1] * 4
    config = Configs(from_begin=True,
                     save_model=True,
                     model_file_name='test_ZEST_zh001.pt',
                     premodel_path='./bert-base-chinese',
                     batch_size=64,
                     learning_rate=1e-4,
                     seed=0,
                     data='../data/mdla/mdla-pisa-train.csv',
                     src_type='mdla_zh',
                     epochs=len(TRAIN_OBJ),
                     train_obj=TRAIN_OBJ,  # 'ce', 'spcl', 'spdcl'
                     shuffle=False,
                     max_utterance_history_len=6,
                     extra_target_ratio=0.5,
                     device='cuda:5',
                     output_dir='./save', )

    spcl_train(config)


if __name__ == "__main__":
    epoch = 19
    epochs = 20
    st = 1 - epoch / epochs
    ed = epoch / epochs
    num_data = 20
    prob_list = [
        (st + (ed - st) / (num_data - 1) * i)*0.6 + 0.4 for i in range(num_data)  # 0.9 + (-0.8)/9999*5 0.8 + (-0.6)/9999*5
    ]
    print(prob_list)
