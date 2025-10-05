#!/bin/bash

ROOT_DIR="datasets/xd-violence"

# timeout in minutes between retryes after 429 error too many requests in 5 minutes
# python XDV_download.py --rootdir $ROOT_DIR --timeout 1 #--test_only
wget -P $ROOT_DIR/test_videos https://huggingface.co/datasets/jherng/xd-violence/resolve/main/data/video/test_videos/Before.Sunset.2004__%2301-14-30_01-16-51_label_A.mp4
wget -P $ROOT_DIR/test_videos https://huggingface.co/datasets/jherng/xd-violence/resolve/main/data/video/test_videos/v%3DwQrV75N2BrI__%231_label_A.mp4


# wget -O XDV_Test.txt https://huggingface.co/datasets/jherng/xd-violence/raw/main/data/test_list.txt
# wget -O XDV_Train.txt https://huggingface.co/datasets/jherng/xd-violence/raw/main/data/train_list.txt


# mkdir -p $ROOT_DIR/test 
# mv $ROOT_DIR/data/video/test_videos/*.mp4 $ROOT_DIR/test_videos
# rm $ROOT_DIR/data/video/test_videos

# rm $ROOT_DIR/.cache
