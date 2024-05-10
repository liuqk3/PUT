python scripts/inference_inpainting.py --name /home/liuqk/Program/python/image-synthesis/OUTPUT/p_vqvae_ffhq_n8192n1024_zeromask0.85_bs24g8/checkpoint/last.pth --image_dir data/inpainting-image-mask/ffhq/gt --gpu 0 --func inference_reconstruction --save_dir RESULT


python scripts/inference_inpainting.py --name /home/liuqk/Program/python/image-synthesis/OUTPUT/p_vqvae_ffhq_n8192n1024_zeromask0.85_res512_bs8g9/checkpoint/000149e_143700iter.pth --image_dir data/inpainting-image-mask/ffhq_512/gt --gpu 0 --func inference_reconstruction --input_res 512,512 --save_dir RESULT
python scripts/inference_inpainting.py --name /home/liuqk/Program/python/image-synthesis/OUTPUT/p_vqvae_ffhq_n8192n1024_zeromask0.85_res512_bs8g9/checkpoint/000149e_143700iter.pth --image_dir data/inpainting-image-mask/ffhq/gt --gpu 0 --func inference_reconstruction --input_res 256,256 --save_dir RESULT/512_test_256




python scripts/inference_inpainting.py --name OUTPUT/p_vqvae_imagenet_n8192_zeromask0.95_f16_bs32g16/checkpoint/000139e_350280iter.pth --image_dir data/inpainting-image-mask/imagenet/gt --gpu 0 --func inference_reconstruction --input_res 256,256 --save_dir RESULT
python scripts/inference_inpainting.py --name OUTPUT/p_vqvae_imagenet_n8192n1024_zeromask0.95_res512_bs24g16/checkpoint/last.pth --image_dir data/inpainting-image-mask/imagenet/gt --gpu 0 --func inference_reconstruction --input_res 256,256 --save_dir RESULT

# mage vqgan
python scripts/inference_inpainting.py --name configs/tools/mage/mage_vqgan.yaml --image_dir data/inpainting-image-mask/imagenet/gt --gpu 0 --func inference_reconstruction --input_res 256,256 --save_dir RESULT