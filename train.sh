python -u train.py --gpu_ids 0 --test_image_path ./validation_images.txt \
       --validation_path ./results/validation_ --test_path ./results/test_ \
       --save_path ./model_save/resnet18_ \
       2>&1 |tee ./log.txt
