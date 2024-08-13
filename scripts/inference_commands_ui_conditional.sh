

python scripts/image_completion_with_ui_conditional.py --name tpami2024_ffhq_256 --num_samples 8  --num_token_per_iter 10 --topk 200 --batch_size 4 --save_dir RESULT/inpainting_with_ui/ffhq_conditional --im_path data/ffhq_256_sample/gt --mask_path data/ffhq_256_sample/mr0.5_0.6 --ui
python scripts/image_completion_with_ui_conditional.py --name tpami2024_naturalscene_256 --num_samples 8  --num_token_per_iter 10 --topk 200 --batch_size 4 --save_dir RESULT/inpainting_with_ui/naturalscene_conditional --im_path data/naturalscene_256_sample/gt --mask_path data/naturalscene_256_sample/mr0.5_0.6 --ui
python scripts/image_completion_with_ui_conditional.py --name tpami2024_imagenet_256 --num_samples 8  --num_token_per_iter 10 --topk 200 --batch_size 4 --save_dir RESULT/inpainting_with_ui/imagenet_conditional --im_path data/imagenet_256_sample/gt --mask_path data/imagenet_256_sample/mr0.5_0.6 --ui
