import copy
import time
from logging import Logger
from sklearn.metrics import cohen_kappa_score
import torch
from tqdm import tqdm
from sklearn import metrics
from .Configs import Configs
from .BertFinetune import BertFinetune
from .utils import get_logger

log: Logger


class Coach:

    def __init__(self, train_data, dev_data, test_data, model: BertFinetune, opt, args: Configs):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.opt = opt
        self.args = args
        self.label_to_idx = args.label_to_index
        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None
        global log
        log = get_logger(name=args.model_file_name[:-3])

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = self.best_dev_f1, self.best_epoch, self.best_state

        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            dev_f1, dev_acc, dev_kappa = self.evaluate()
            log.info("[Dev set] [f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}]".format(dev_f1, dev_acc, dev_kappa))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")
            test_f1, test_acc, test_kappa = self.evaluate(test=True)
            log.info("[Test set] [f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}]".format(test_f1, test_acc, test_kappa))

        # The best
        log.info("Best in epoch {}, best_dev_f1: {}".format(best_epoch, best_dev_f1))
        self.model.load_state_dict(best_state)
        dev_f1, dev_acc, dev_kappa = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}]".format(dev_f1, dev_acc, dev_kappa))
        test_f1, test_acc, test_kappa = self.evaluate(test=True)
        log.info("[Test set] [f1 {:.4f}] [acc {:.4f}] [kappa {:.4f}]".format(test_f1, test_acc, test_kappa))

        return best_dev_f1, best_epoch, best_state

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()

        for idx, batch in tqdm(enumerate(self.train_data.dataloader), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            loss = self.model.get_loss(batch)
            epoch_loss += loss.item()
            loss.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False):
        data = self.test_data if test else self.dev_data
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx, batch in tqdm(enumerate(data.dataloader), desc="test" if test else "dev"):
                golds.append(batch["labels_tensor"].to("cpu"))
                y_hat = self.model(batch)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
            kappa = cohen_kappa_score(golds, preds)

        return f1, acc, kappa