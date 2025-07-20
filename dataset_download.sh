ROOT_DIR=/mnt/datasets/ucf_crime

wget -P $ROOT_DIR https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/Anomaly-Videos-Part-1.zip
wget -P $ROOT_DIR https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/Anomaly-Videos-Part-2.zip
wget -P $ROOT_DIR https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/Anomaly-Videos-Part-3.zip
wget -P $ROOT_DIR https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/Anomaly-Videos-Part-4.zip
wget -P $ROOT_DIR https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/Testing_Normal_Videos.zip
wget -P $ROOT_DIR https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/Training-Normal-Videos-Part-1.zip
wget -P $ROOT_DIR https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/Training-Normal-Videos-Part-2.zip
# wget -P $ROOT_DIR https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Test.txt
# wget -P $ROOT_DIR https://huggingface.co/datasets/jinmang2/ucf_crime/resolve/main/UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Train.txt

# unzip "$ROOT_DIR/Anomaly-Videos-Part-1.zip" -d "$ROOT_DIR"
# rm "$ROOT_DIR/Anomaly-Videos-Part-1.zip"

# unzip "$ROOT_DIR/Anomaly-Videos-Part-2.zip" -d "$ROOT_DIR"
# rm "$ROOT_DIR/Anomaly-Videos-Part-2.zip"

# unzip "$ROOT_DIR/Anomaly-Videos-Part-3.zip" -d "$ROOT_DIR"
# rm "$ROOT_DIR/Anomaly-Videos-Part-3.zip"

# unzip "$ROOT_DIR/Anomaly-Videos-Part-4.zip" -d "$ROOT_DIR"
# rm "$ROOT_DIR/Anomaly-Videos-Part-4.zip"

# unzip "$ROOT_DIR/Training-Normal-Videos-Part-1.zip" -d "$ROOT_DIR"
# rm "$ROOT_DIR/Training-Normal-Videos-Part-1.zip"

# unzip "$ROOT_DIR/Training-Normal-Videos-Part-2.zip" -d "$ROOT_DIR"
# rm "$ROOT_DIR/Training-Normal-Videos-Part-2.zip"

# unzip "$ROOT_DIR/Testing_Normal_Videos.zip" -d "$ROOT_DIR"
# rm "$ROOT_DIR/Testing_Normal_Videos.zip"


# mv $ROOT_DIR/Anomaly-Videos-Part-1/*/ $ROOT_DIR/
# rm -r $ROOT_DIR/Anomaly-Videos-Part-1

# mv $ROOT_DIR/Anomaly-Videos-Part-2/*/ $ROOT_DIR/
# rm -r $ROOT_DIR/Anomaly-Videos-Part-2

# mv $ROOT_DIR/Anomaly-Videos-Part-3/*/ $ROOT_DIR/
# rm -r $ROOT_DIR/Anomaly-Videos-Part-3

# mv $ROOT_DIR/Anomaly-Videos-Part-4/*/ $ROOT_DIR/
# rm -r $ROOT_DIR/Anomaly-Videos-Part-4

