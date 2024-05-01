To perform testing:

python test.py --content "D:/finalyearproject/project/test_images/COCO_train2014_000000000025.jpg" --style "D:/finalyearproject/project/test_images/adriaen-brouwer_dune-landscape-by-moonlight.jpg" --output "D:/finalyearproject/project/output"

To perform Training:

python train.py --style_dir ./datasets/Images --content_dir ./datasets/train2014 --save_dir ./experiments --max_iter 5000 --batch_size 2 --n_threads 0 --save_model_interval 200