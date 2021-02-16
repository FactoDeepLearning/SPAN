#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

from basic.generic_training_manager import GenericTrainingManager
from basic.utils import edit_wer_from_list, nb_words_from_list, nb_chars_from_list,  LM_ind_to_str
import torch
from torch.nn import CTCLoss
import editdistance
from torch.cuda.amp import autocast


class TrainerPGCTC(GenericTrainingManager):

    def __init__(self, params):
        super(TrainerPGCTC, self).__init__(params)

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def train_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        y_len = batch_data["labels_len"]
        str_y = batch_data["raw_labels"]
        loss = 0

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="mean")
        self.optimizer.zero_grad()

        with autocast(enabled=self.params["training_params"]["use_amp"]):
            global_pred = self.models["decoder"](self.models["encoder"](x))

            ind_x = list()
            b, c, h, w = global_pred.size()

            for i in range(b):
                x_h, x_w = batch_data["imgs_reduced_shape"][i][:2]
                pred = global_pred[i, :, :x_h, :x_w]
                pred = pred.reshape(1, c, x_h*x_w)
                torch.backends.cudnn.enabled = False
                loss += loss_ctc(pred.permute(2, 0, 1), y[i].unsqueeze(0), [x_h*x_w, ], [y_len[i], ])
                torch.backends.cudnn.enabled = True
                ind_x.append(torch.argmax(pred, dim=1).cpu().numpy()[0])

        del global_pred
        self.backward_loss(loss)
        self.step_optimizer()
        metrics = self.compute_metrics(ind_x, str_y, loss=loss.item(), metric_names=metric_names)
        return metrics

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        y_len = batch_data["labels_len"]
        str_y = batch_data["raw_labels"]
        loss = 0

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"], reduction="mean")
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)
            global_pred = self.models["decoder"](x)

            ind_x = list()
            b, c, h, w = global_pred.size()
            for i in range(b):
                x_h, x_w = batch_data["imgs_reduced_shape"][i][:2]
                pred = global_pred[i, :, :x_h, :x_w]
                pred = pred.reshape(1, c, x_h*x_w)
                loss += loss_ctc(pred.permute(2, 0, 1), y[i].unsqueeze(0), [x_h*x_w, ], [y_len[i], ])
                ind_x.append(torch.argmax(pred, dim=1).cpu().numpy()[0])

        metrics = self.compute_metrics(ind_x, str_y, loss=loss.item(), metric_names=metric_names)
        if "pred" in metric_names:
            metrics["pred"].extend([batch_data["unchanged_labels"], batch_data["names"]])
        return metrics

    def compute_metrics(self, ind_x, str_y, loss=None, metric_names=list()):
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
        metrics = dict()
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u,v in zip(str_y, str_x)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
            elif metric_name == "wer":
                metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                metrics["nb_words"] = nb_words_from_list(str_y)
            elif metric_name == "pred":
                metrics["pred"] = [str_x, ]
        if "loss_ctc" in metric_names:
            metrics["loss_ctc"] = loss
        metrics["nb_samples"] = len(str_y)
        return metrics

