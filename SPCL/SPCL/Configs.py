import torch
import torch.nn.functional as f
from transformers import AutoTokenizer

_Label_To_Index = {'A1': 0, 'B1': 1, 'C1': 2, 'D1': 3,
                  'A2': 4, 'B2': 5, 'C2': 6, 'D2': 7,
                  'A3': 8, 'B3': 9, 'C3': 10, 'D3': 11,
                  'U': 12}

def _score_func(x, y):
    return (1 + f.cosine_similarity(x, y, dim=-1)) / 2 + 1e-8


class Configs(object):
    def __init__(self,
                 # Training parameters
                 from_begin: bool = True,
                 model_file_name: str = "model.pt",
                 save_model: bool = False,

                 device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu").type,
                 data: str = '../data/data.pkl',
                 names_pool_dir: str = '../data/names_pool.json',
                 epochs: int = 10,
                 batch_size: int = 4,
                 optimizer: str = "adam",
                 learning_rate: float = 1e-3,
                 # weight_decay: float = 1e-8,
                 # max_grad_value: float = -1,
                 # drop_rate: float = 0.5,
                 max_sentence_len=256,  # 考虑的对话历史，单词数限长
                 max_utterance_history_len=8,  # 考虑的对话历史，句子数限长
                 extra_target_ratio=0.4,  # 对话历史中每一个额外目标被选择的概率
                 # SupProtoConLoss
                 temperature=0.08,
                 pool_size=512,
                 support_set_size=64,
                 train_obj='ce',
                 cl_selection_ratio=1,  # course learning 选择数据的比例
                 cl=True,
                 accumulation_steps=1,
                 fgm=False,
                 warm_up=128,
                 ptmlr= 1e-5,

                 # Model parameters
                 # wp: int = 2,  # Past context window size. Set wp to -1 to use all the past context.
                 # wf: int = 2,  # Future context window size. Set wp to -1 to use all the future context.
                 premodel_path: str = 'bert-base-uncased',  # 'bert-base-uncased'
                 # n_speakers: int = 3,
                 feature_dim: int = 768,  # 模型输入的初始数据中文本特征向量的维度
                 hidden_size: int = None,
                 # class_weight: torch.Tensor = None,
                 num_classes: int = len(_Label_To_Index),
                 label_to_index: dict = _Label_To_Index,

                 # others
                 seed: int = 1,
                 shuffle: bool = True,
                 log_file_name: str = 'log.txt',
                 output_dir: str = "./save",
                 score_func = _score_func
                 ):
        # Training parameters
        self.from_begin = from_begin
        self.model_file_name = model_file_name
        self.save_model = save_model

        self.device = device
        self.data = data
        self.names_pool_dir = names_pool_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        # self.weight_decay = weight_decay
        # self.max_grad_value = max_grad_value
        # self.drop_rate = drop_rate
        self.max_sentence_len = max_sentence_len
        self.max_utterance_history_len = max_utterance_history_len
        self.extra_target_ratio = extra_target_ratio
        self.temperature = temperature
        self.pool_size = pool_size
        self.support_set_size = support_set_size
        self.train_obj = train_obj
        self.cl_selection_ratio = cl_selection_ratio
        self.cl = cl
        self.accumulation_steps = accumulation_steps
        self.fgm = fgm
        self.warm_up = warm_up
        self.ptmlr = ptmlr

        # Model parameters
        self.premodel_path = premodel_path
        # self.n_speakers = n_speakers
        self.feature_dim = feature_dim  # 模型输入的初始数据中文本特征向量的维度
        self.hidden_size = hidden_size
        # self.class_weight = class_weight
        self.num_classes = num_classes
        self.label_to_index = label_to_index

        # others
        self.seed = seed
        self.shuffle = shuffle
        self.log_file_name = log_file_name
        self.output_dir = output_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.premodel_path, local_files_only=True)
        _special_tokens_ids = self.tokenizer('<mask>')['input_ids']
        self.CLS = _special_tokens_ids[0]
        self.MASK = _special_tokens_ids[1]
        self.SEP = _special_tokens_ids[2]
        self.pad_value = 1
        self.score_func = score_func
