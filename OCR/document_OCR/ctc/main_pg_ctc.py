#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to automatically recognize text from text-block images following a single-column layout.
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

import os
import sys
from os.path import dirname
DOSSIER_COURRANT = dirname(os.path.abspath(__file__))
ROOT_FOLDER = dirname(dirname(dirname(DOSSIER_COURRANT)))
sys.path.append(ROOT_FOLDER)
from OCR.document_OCR.ctc.trainer_pg_ctc import TrainerPGCTC
from basic.models import FCN_Encoder
from OCR.document_OCR.ctc.models_pg_ctc import Decoder
from torch.optim import Adam
from basic.generic_dataset_manager import OCRDataset
import torch.multiprocessing as mp
import torch


def train_and_test(rank, params):
    params["training_params"]["ddp_rank"] = rank
    model = TrainerPGCTC(params)
    # Model is trained until max_time_training or max_nb_epochs is reached
    model.train()

    ## load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()


    ## compute metrics on train, valid and test sets (in eval conditions)
    metrics = ["cer", "wer"]
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["train", "valid", "test"]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics, output=True)


if __name__ == "__main__":
    dataset_name = "IAM"
    dataset_level = "paragraph"

    params = {
        "dataset_params": {
            "datasets": {
                dataset_name: "../../../Datasets/formatted/{}_{}".format(dataset_name, dataset_level),
            },
            "train": {
                "name": "{}-train".format(dataset_name),
                "datasets": [dataset_name, ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [dataset_name, ],
            },
            "dataset_class": OCRDataset,
            "config": {
                "width_divisor": 8,  # Image width will be divided by 8
                "height_divisor": 32,  # Image height will be divided by 32
                "padding_value": 0,  # Image padding value
                "padding_token": 1000,  # Label padding value
                "charset_mode": "ctc",  # add blank token
                "constraints": ["CTC_pg", "normalize"],  # handle minimum width > 2* len(label)
                "preprocessings": [
                    {
                        "type": "dpi",  # modify image resolution
                        "source": 300,  # from 300 dpi
                        "target": 150,  # to 150 dpi
                    },
                    {
                        "type": "to_RGB",  # if grayscaled image, produce RGB one (3 channels with same value) otherwise do nothing
                    },
                ],
                "augmentation": {
                    "dpi": {
                        "proba": 0.2,
                        "min_factor": 0.75,
                        "max_factor": 1,
                    },
                    "perspective": {
                        "proba": 0.2,
                        "min_factor": 0,
                        "max_factor": 0.3,
                    },
                    "elastic_distortion": {
                        "proba": 0.2,
                        "max_magnitude": 20,
                        "max_kernel": 3,
                    },
                    "random_transform": {
                        "proba": 0.2,
                        "max_val": 125,
                    },
                    "dilation_erosion": {
                        "proba": 0.2,
                        "min_kernel": 1,
                        "max_kernel": 3,
                        "iterations": 1,
                    },
                    "brightness": {
                        "proba": 0.2,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "contrast": {
                        "proba": 0.2,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "sign_flipping": {
                        "proba": 0.2,
                    },
                },
            }
        },

        "model_params": {
            "models": {
                "encoder": FCN_Encoder,
                "decoder": Decoder
            },
            # "transfer_learning": None,
            "transfer_learning": {
                # model_name: [state_dict_name, checkpoint_path, learnable, strict]
                "encoder": ["encoder", "../../../Pool-Line-R_iam.pt", True, True],
            },
            "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
            "dropout": 0.5,

        },

        "training_params": {
            "output_folder": "SPAN-PT-RA_iam",
            "max_nb_epochs": 5000,
            "max_training_time":  3600*48,  # 2J limit
            "load_epoch": "last",  # ["best", "last"] to continue training
            "interval_save_weights": None,  # None: keep best and last only
            "use_ddp": False,  # Use DistributedDataParallel
            "use_amp": False,  # Enable automatic mix-precision
            "nb_gpu": torch.cuda.device_count(),
            "batch_size": 8,  # batch size per GPU
            "optimizer": {
                "class": Adam,
                "args": {
                    "lr": 0.0001,
                    "amsgrad": False,
                }
            },
            "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
            "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
            "focus_metric": "cer",  # Metrics to focus on to determine best epoch
            "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
            "set_name_focus_metric": "{}-valid".format(dataset_name),
            "train_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for training
            "eval_metrics": ["loss_ctc", "cer", "wer"],  # Metrics name for evaluation on validation set during training
            "force_cpu": False,  # True for debug purposes to run on cpu only
        },
    }

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params)

