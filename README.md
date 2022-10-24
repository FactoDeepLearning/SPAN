# SPAN: a Simple Predict & Align Network for Handwritten Paragraph Recognition
This repository is a public implementation of the paper: "SPAN: a Simple Predict & Align Network forHandwritten Paragraph Recognition".

To discover my other works, here is my [academic page](https://factodeeplearning.github.io/).

The paper is available at https://arxiv.org/abs/2102.08742.

It focuses on Optical Character Recognition (OCR) applied at line and paragraph levels.

We obtained the following results at line level:

|  Dataset  |  cer |  wer  |
|:------------:|:----:|:-----:|
|      IAM     | 4.82 | 18.17 |
|     RIMES    | 3.02 | 10.73 |
|   READ 2016   | 4.56 | 21.07 |

For the paragraph level, here are the results:

| Dataset  |  cer |  wer  |
|:------------:|:----:|:-----:|
|      IAM     | 5.45 | 19.83 |
|     RIMES    | 4.17 | 15.61 |
|   READ 2016   | 6.20 | 25.69 |

Pretrained model weights are available [here](https://git.litislab.fr/dcoquenet/span) and [here](https://zenodo.org/record/7244546).

Table of contents:
1. [Getting Started](#Getting-Started)
2. [Datasets](#Datasets)
3. [Training And Evaluation](#Training-and-evaluation)

## Getting Started
Implementation has been tested with Python 3.7, torch 1.7.1 and CUDA 11.

Clone the repository:

```
git clone https://github.com/FactoDeepLearning/SPAN.git
```

Install the dependencies:

```
pip install -r requirements.txt
```


## Datasets
This section is dedicated to the datasets used in the paper: download and formatting instructions are provided 
for experiment replication purposes.

### IAM

#### Details

IAM corresponds to english grayscale handwriting images (from the LOB corpus).
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 6,482 |     976    | 2,915 |
| paragraph |  747  |     116    |  336  |

#### Download



- Register at the [FKI's webpage](http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php)
- Download the dataset [here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database) 
- Move the following files into the folder Datasets/raw/IAM/
    - formsA-D.tgz
    - formsE-H.tgz
    - formsI-Z.tgz
    - lines.tgz
    - ascii.tgz



### RIMES

#### Details

RIMES corresponds to french grayscale handwriting images.
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 9,947 |     1,333  | 778 |
| paragraph |  1400 |     100    |  100 |

#### Download

- Fill in the a2ia user agreement form available [here](http://www.a2ialab.com/doku.php?id=rimes_database:start) and send it by email to rimesnda@a2ia.com. You will receive by mail a username and a password
- Login in and download the data from [here](http://www.a2ialab.com/doku.php?id=rimes_database:data:icdar2011:line:icdar2011competitionline)
- Move the following files into the folder Datasets/raw/RIMES/
    - eval_2011_annotated.xml
    - eval_2011_gray.tar
    - training_2011_gray.tar
    - training_2011.xml

### READ 2016

#### Details
READ 2016 corresponds to Early Modern German RGB handwriting images.
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 8,349 |  1,040    | 1,138|
| paragraph |  1584 |     179    | 197 |

#### Download

- From root folder:

```
cd Datasets/raw
mkdir READ_2016
cd READ_2016
wget https://zenodo.org/record/1164045/files/{Test-ICFHR-2016.tgz,Train-And-Val-ICFHR-2016.tgz}
```


### Format the datasets

- Comment/Uncomment the following lines from the main function of the script "format_datasets.py" according to your needs and run it

```
if __name__ == "__main__":

    # format_IAM_line()
    # format_IAM_paragraph()

    # format_RIMES_line()
    # format_RIMES_paragraph()

    # format_READ2016_line()
    # format_READ2016_paragraph()

```

- This will generate well-formated datasets, usable by the training scripts.


## Training And Evaluation
You need to have a properly formatted dataset to train a model, please refer to the section [Datasets](#Datasets). 

Two scripts are provided to train respectively line and paragraph level models: OCR/line_OCR/ctc/main_line_ctc.py and OCR/document_OCR/ctc/main_pg_ctc.py

Training a model leads to the generation of output files ; they are located in the output folder OCR/line_OCR/ctc/outputs/#TrainingName or OCR/document_OCR/ctc/outputs/#TrainingName.

The outputs files are split into two subfolders: "checkpoints" and "results". "checkpoints" contains model weights for the last trained epoch and for the epoch giving the best valid CER.
"results" contains tensorboard log for loss and metrics as well as text file for used hyperparameters and results of evaluation.

Training can use automatic mix-precision.

All hyperparameters are specified and editable in the training scripts (meaning are in comments).

Evaluation is performed just after training ending (training is stopped when the maximum elapsed time is reached or after a maximum number of epoch as specified in the training script)

## Citation
```bibtex
@inproceedings{Coquenet2021,
    author    = {Denis Coquenet and
        Cl{\'{e}}ment Chatelain and
        Thierry Paquet},
        title     = {SPAN: {A} Simple Predict {\&} Align Network for Handwritten
        Paragraph Recognition},
        booktitle = {16th International Conference on Document Analysis and Recognition,
            {ICDAR}},
            series    = {Lecture Notes in Computer Science},
            volume    = {12823},
            pages     = {70--84},
            year      = {2021},
            doi       = {10.1007/978-3-030-86334-0\_5},

        }
```

## License

This whole project is under Cecill-C license EXCEPT FOR the file "basic/transforms.py" which is under MIT license.
