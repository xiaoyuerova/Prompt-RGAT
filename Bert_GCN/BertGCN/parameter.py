import torch

Label_To_Index_PISA = {'A1': 0, 'B1': 1, 'C1': 2, 'D1': 3,
                       'A2': 4, 'B2': 5, 'C2': 6, 'D2': 7,
                       'A3': 8, 'B3': 9, 'C3': 10, 'D3': 11,
                       'U': 12}

Label_To_Index_AJ = {'SESU': 0, 'SMC': 1, 'SN': 2, 'SSI': 3,
                     'CRF': 4, 'CP': 5, 'CEC': 6, 'CMC': 7}


class Parameter(object):
    def __init__(self,
                 from_begin: bool = True,
                 model_file_name: str = "model.pt",
                 output_dir: str = "./save",
                 save_model: bool = True,
                 device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu").type,
                 data: str = './data/data.pkl',
                 src_type: str = 'mdla_pisa',  # 蒙德里安pisa：'mdla_pisa' or 三电阻：'sdz_pisa' or 三电阻: 'sdz_aj'
                 epochs: int = 10,
                 batch_size: int = 4,
                 optimizer: str = "adam",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-8,
                 max_grad_value: float = -1,
                 drop_rate: float = 0.5,

                 # Model parameters
                 wp: int = 2,  # Past context window size. Set wp to -1 to use all the past context.
                 wf: int = 2,  # Future context window size. Set wp to -1 to use all the future context.
                 n_speakers: int = 3,
                 hidden_size: int = None,
                 class_weight: torch.Tensor = None,
                 label_to_index: dict = None,

                 # others
                 seed: int = 1,
                 text_dim: int = 768  # 模型输入的初始数据中文本特征向量的维度
                 ):
        # Training parameters
        self.from_begin = from_begin
        self.model_file_name = model_file_name
        self.output_dir = output_dir
        self.save_model = save_model
        self.device = device
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_value = max_grad_value
        self.drop_rate = drop_rate

        # Model parameters
        self.wp = wp
        self.wf = wf
        self.n_speakers = n_speakers
        self.hidden_size = hidden_size
        self.class_weight = class_weight
        if label_to_index is None:
            if src_type == 'mdla_pisa' or src_type == 'sdz_pisa':
                self.label_to_index = Label_To_Index_PISA
            else:
                self.label_to_index = Label_To_Index_AJ
        self.tag_size = len(self.label_to_index)

        # others
        self.seed = seed
        self.text_dim = text_dim  # 模型输入的初始数据中文本特征向量的维度

        # for k, v in kwargs.items():
        #     if k in self.__dict__:  # 只接受预定属性的值
        #         setattr(self, k, v)
