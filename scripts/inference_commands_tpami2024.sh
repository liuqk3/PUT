##########################
# ffhq, resolution 256x256
##########################

# sample 20 patches from top 200 tokens in one iteration, only one image is generated 
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_ffhq_res256/checkpoint/last.pth \
--input_res 256,256 \
--num_token_per_iter 20 \
--num_token_for_sampling 200 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/ffhq_256_sample/gt \
--mask_dir data/ffhq_256_sample/mr0.5_0.6 \
--gpu 0

# sample all patches from top 1 token in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_ffhq_res256/checkpoint/last.pth \
--input_res 256,256 \
--num_token_per_iter -1 \
--num_token_for_sampling 1 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/ffhq_256_sample/gt \
--mask_dir data/ffhq_256_sample/mr0.5_0.6 \
--gpu 0


##########################
# ffhq, resolution 512x512
##########################
# sample 20 patches from top 200 tokens in one iteration, only one image is generated 
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_ffhq_res512/checkpoint/last.pth \
--input_res 512,512 \
--num_token_per_iter 20 \
--num_token_for_sampling 200 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/ffhq_512_sample/gt \
--mask_dir data/ffhq_512_sample/mr0.5_0.6 \
--gpu 0

# sample all patchs from top 1 token in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_ffhq_res512/checkpoint/last.pth \
--input_res 512,512 \
--num_token_per_iter -1 \
--num_token_for_sampling 1 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/ffhq_512_sample/gt \
--mask_dir data/ffhq_512_sample/mr0.5_0.6 \
--gpu 0






########################
# naturalscene, 256x256
########################

# sample 20 patches from top 200 tokens in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_naturalscene_res256/checkpoint/000599e_742800iter.pth \
--input_res 256,256 \
--num_token_per_iter 20 \
--num_token_for_sampling 200 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/naturalscene_256_sample/gt \
--mask_dir data/naturalscene_256_sample/mr0.5_0.6 \
--gpu 0

# sample all patches from top 200 token in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_naturalscene_res256/checkpoint/000599e_742800iter.pth \
--input_res 256,256 \
--num_token_per_iter -1 \
--num_token_for_sampling 200 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/naturalscene_256_sample/gt \
--mask_dir data/naturalscene_256_sample/mr0.5_0.6 \
--gpu 0



########################
# naturalscene, 512x512
########################

# sample 20 patches from top 200 tokens in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_naturalscene_res512/checkpoint/000599e_742800iter.pth \
--input_res 512,512 \
--num_token_per_iter 20 \
--num_token_for_sampling 200 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/naturalscene_512_sample/gt \
--mask_dir data/naturalscene_512_sample/mr0.5_0.6 \
--gpu 0

# sample all patches from top 200 token in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_naturalscene_res512/checkpoint/000599e_742800iter.pth \
--input_res 512,512 \
--num_token_per_iter -1 \
--num_token_for_sampling 200 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/naturalscene_512_sample/gt \
--mask_dir data/naturalscene_512_sample/mr0.5_0.6 \
--gpu 0






########################
# imagenet, 256x256
########################

# sample 20 patches from top 200 tokens in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_imagenet_res256/checkpoint/000299e_1000799iter.pth \
--input_res 256,256 \
--num_token_per_iter 20 \
--num_token_for_sampling 200 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/imagenet_256_sample/gt \
--mask_dir data/imagenet_256_sample/mr0.5_0.6 \
--gpu 0

# sample all patches from top 1 token in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_imagenet_res256/checkpoint/000299e_1000799iter.pth \
--input_res 256,256 \
--num_token_per_iter -1 \
--num_token_for_sampling 1 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/imagenet_256_sample/gt \
--mask_dir data/imagenet_256_sample/mr0.5_0.6 \
--gpu 0


########################
# imagenet, 512x512
########################

# sample 20 patches from top 200 tokens in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_imagenet_res512/checkpoint/last.pth \
--input_res 512,512 \
--num_token_per_iter 20 \
--num_token_for_sampling 200 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/imagenet_512_sample/gt \
--mask_dir data/imagenet_512_sample/mr0.5_0.6 \
--gpu 0

# sample all patches from top 1 token in one iteration
python scripts/inference.py --func inference_inpainting \
--name OUTPUT/tpami2024_vit_base_imagenet_res512/checkpoint/last.pth \
--input_res 512,512 \
--num_token_per_iter -1 \
--num_token_for_sampling 1 \
--num_sample 1 \
--num_replicate 1 \
--image_dir data/imagenet_512_sample/gt \
--mask_dir data/imagenet_512_sample/mr0.5_0.6 \
--gpu 0