#!/bin/bash


function download_zip {
    gdown "${GDRIVE_LINK}" -O "${zip_fname}"
    unzip "${zip_fname}"
    rm "${zip_fname}"
}


#### Download unofficial model checkpoints. ####
GDRIVE_LINK="https://drive.google.com/uc?id=1w_0aIgBCQIsDqdO4HKBKkqNpFJvm7O4z"
# Download and unzip the following ./logs/bc/train/fold.
# - Fold Cloth model checkpoints.
zip_fname="fold_checkpoints.zip"
download_zip


#### Download official model checkpoints. ####


GDRIVE_LINK="https://drive.google.com/uc?id=1iVzG5kdMOkFSiRnkXDqdje2fwh38cXP9"
# Download and unzip the following ./logs/bc/train/pusht.
# - PushT model checkpoints.
zip_fname="pusht_checkpoints.zip"
download_zip


GDRIVE_LINK="https://drive.google.com/uc?id=1oDmWrQY8UbYKn1Fjh37ExaHXZtPatij_"
# Download and unzip the following ./logs/bc/train/close.
# - Close Box model checkpoints.
zip_fname="close_checkpoints.zip"
download_zip


GDRIVE_LINK="https://drive.google.com/uc?id=1A2Y2wPBqn5m8D0UnTIkj1svbzE4wAWSA"
# Download and unzip the following ./logs/bc/train/cover.
# - Cover Object model checkpoints.
zip_fname="cover_checkpoints.zip"
download_zip


GDRIVE_LINK="https://drive.google.com/uc?id=10Vd0Bf6OHTV_9r7mGFa558J7baYzTyOT"
# Download and unzip the following ./logs/bc/train/push_chair.
# - Push Chair model checkpoints.
zip_fname="push_chair_checkpoints.zip"
download_zip