
python scripts/inference_inpainting.py --func inference_inpainting \
--name  OUTPUT/transformer_imagenet/checkpoint/000044e_600524iter.pth \
--input_res 256,256 \
--num_token_per_iter 2,5,10 --num_token_for_sampling 50,100 \
--image_dir data/inpainting-image-mask/imagenet/gt \
--mask_dir data/inpainting-image-mask/imagenet/mr0.1_0.6 \
--save_masked_image \
--save_dir mr0.1_0.6 \
--gpu 0