#!/bin/bash


function download_zip {
    gdown "${GDRIVE_LINK}" -O "${zip_fname}"
    unzip "${zip_fname}"
    rm "${zip_fname}"
}


#### Download archived datasets. ####


GDRIVE_LINK="https://drive.google.com/uc?id=1nDeFu2DfEzV8fHeqvdXfRwynY_jcW7q-"
# Download and unzip the following ./logs/bc/eval.
# - Close hyperparameter sweep datasets.
# - Close, Cover visualization datasets.
zip_fname="archived_datasets.zip"
download_zip


#### Download official result datasets. ####


GDRIVE_LINK="https://drive.google.com/uc?id=1wrh9kb7_zcgVzrDFQ5o6BuWhOuWKS5oD"
# Download and unzip the following ./logs/bc/eval.
# - PushT main result (Figure 5): All methods sans VLM.
# - PushT ablation result (Figure 8, Figure 9): STAC.
zip_fname="pusht_datasets.zip"
download_zip


GDRIVE_LINK="https://drive.google.com/uc?id=1yEaLn6P27TvkqstHShPtgjJUkkqY3-dH"
# Download and unzip the following ./logs/bc/eval.
# - Close erratic failures (Table 1, Table 5): All methods sans VLM.
# - Close erratic failures (Table 1, Table 5): STAC and VLM (Sentinel).
# - Close task progression failures (Figure 6, Table 7): STAC and VLM (Sentinel).
zip_fname="close_datasets.zip"
download_zip


GDRIVE_LINK="https://drive.google.com/uc?id=1_TTRCIZHQDsmxh6h3CT0HcU9GzZ4p1ug"
# Download and unzip the following ./logs/bc/eval.
# - Cover task progression failures (Figure 6, Table 6): STAC and VLM (Sentinel).
zip_fname="cover_datasets.zip"
download_zip


GDRIVE_LINK="https://drive.google.com/uc?id=1UW2lxqQiu90NTDsPmqxx8wVZ7aSlp504"
# Download and unzip the following ./logs/bc/real_eval.
# - Push chair real-world result (Table 2): STAC and VLM (Sentinel).
zip_fname="push_chair_datasets.zip"
download_zip
