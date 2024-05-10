########################
# ffhq
########################

# sample one patch from top 50 tokens in one iteration, only one 
python scripts/inference.py --func inference_inpainting \
--name  OUTPUT/cvpr2022_transformer_ffhq/checkpoint/last.pth \
--input_res 256,256 \
--num_token_per_iter 1 \
--num_token_for_sampling 50 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/ffhq_256_sample/gt \
--mask_dir data/ffhq_256_sample/mr0.5_0.6 \
--gpu 0

# sample all patches from top 1 token in one iteration
python scripts/inference.py --func inference_inpainting \
--name  OUTPUT/cvpr2022_transformer_ffhq/checkpoint/last.pth \
--input_res 256,256 \
--num_token_per_iter -1 \
--num_token_for_sampling 1 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/ffhq_256_sample/gt \
--mask_dir data/ffhq_256_sample/mr0.5_0.6 \
--gpu 0


########################
# naturalscene
########################

# sample one patch from top 50 tokens in one iteration
python scripts/inference.py --func inference_inpainting \
--name  OUTPUT/cvpr2022_transformer_naturalscene/checkpoint/000199e_495399iter.pth \
--input_res 256,256 \
--num_token_per_iter 1 \
--num_token_for_sampling 50 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/naturalscene_256_sample/gt \
--mask_dir data/naturalscene_256_sample/mr0.5_0.6 \
--gpu 0

# sample all patches from top 1 token in one iteration
python scripts/inference.py --func inference_inpainting \
--name  OUTPUT/cvpr2022_transformer_naturalscene/checkpoint/000199e_495399iter.pth \
--input_res 256,256 \
--num_token_per_iter -1 \
--num_token_for_sampling 1 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/naturalscene_256_sample/gt \
--mask_dir data/naturalscene_256_sample/mr0.5_0.6 \
--gpu 0



########################
# imagenet
########################

# sample one patch from top 50 tokens in one iteration
python scripts/inference.py --func inference_inpainting \
--name  OUTPUT/cvpr2022_transformer_imagenet/checkpoint/000044e_600524iter.pth \
--input_res 256,256 \
--num_token_per_iter 1 \
--num_token_for_sampling 50 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/imagenet_256_sample/gt \
--mask_dir data/imagenet_256_sample/mr0.5_0.6 \
--gpu 0

# sample all patches from top 1 token in one iteration
python scripts/inference.py --func inference_inpainting \
--name  OUTPUT/cvpr2022_transformer_imagenet/checkpoint/000044e_600524iter.pth \
--input_res 256,256 \
--num_token_per_iter -1 \
--num_token_for_sampling 1 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/imagenet_256_sample/gt \
--mask_dir data/imagenet_256_sample/mr0.5_0.6 \
--gpu 0