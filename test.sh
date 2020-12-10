python -u test.py --gpu_ids 0 --test_image_path ./validation_images.txt --test_path ./test_ \
       --model_path ./model_save/resnet18.pth \
       2>&1 |tee ./log.txt
