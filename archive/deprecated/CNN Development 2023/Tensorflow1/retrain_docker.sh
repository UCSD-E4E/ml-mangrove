read -p 'Model Version: ' version
 
read -p 'Number of Epochs: ' epochs
 
read -p 'Learning Rate: ' learning_rate
 
read -p 'Batch Size: ' batch_size
 
python3 cnn_retrain2.py \
    --image_dir=/mnt/input \
    --output_graph=/mnt/output/output_graph_v$version.pb \
    --output_labels=/mnt/output/output_labels_v$version.pb \
    --how_many_training_steps=$epochs \
    --learning_rate=$learning_rate \
    --train_batch_size=$batch_size \
    --flip_left_right \
    --random_crop=10 \
    --random_scale=10 \
    --random_brightness=10
