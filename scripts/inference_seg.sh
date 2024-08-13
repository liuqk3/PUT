# this scripts is used to get the segmentation masks from gt images and inpainted images. Then the masks are used to mIoU in Table 3.
# inpainted images should be generated before.

python scripts/inference_seg.py --image_dir data/inpainting-image-mask/naturalscene/gt --save_root RESULT/places_seg/gt --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_ffhq_seg_sketch_dual_encoder_res256_e599/places_e_10_60_top200_nTpi-1_numSample1 --save_root RESULT/places_seg/e --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_ffhq_seg_sketch_dual_encoder_res256_e599/places_s_10_60_top200_nTpi-1_numSample1 --save_root RESULT/places_seg/s --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_ffhq_seg_sketch_dual_encoder_res256_e599/places_s_e_10_60_top200_nTpi-1_numSample1 --save_root RESULT/places_seg/s_e --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_ffhq_seg_sketch_dual_encoder_res256_e599/places_none_10_60_top200_nTpi-1_numSample1 --save_root RESULT/places_seg/none --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_ffhq_seg_sketch_dual_encoder_res256_e599/places_10_60_top200_nTpi-1_numSample1 --save_root RESULT/places_seg/train_no_cond --gpu 0

# python scripts/inference_seg.py --image_dir data/inpainting-image-mask/ffhq/gt --save_root RESULT/ffhq_seg/gt --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/vit768_ffhq_seg_sketch_dual_encoder_e799/ffhq_e_10_60_top1_nTpi-1_numSample1 --save_root RESULT/ffhq_seg/e --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/vit768_ffhq_seg_sketch_dual_encoder_e799/ffhq_s_10_60_top1_nTpi-1_numSample1 --save_root RESULT/ffhq_seg/s --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/vit768_ffhq_seg_sketch_dual_encoder_e799/ffhq_s_e_10_60_top1_nTpi-1_numSample1 --save_root RESULT/ffhq_seg/s_e --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/vit768_ffhq_seg_sketch_dual_encoder_e799/ffhq_none_10_60_top1_nTpi-1_numSample1 --save_root RESULT/ffhq_seg/none --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/vit768_ffhq_seg_sketch_dual_encoder_e799/ffhq_10_60_top1_nTpi-1_numSample1 --save_root RESULT/ffhq_seg/train_no_cond --gpu 0

# python scripts/inference_seg.py --image_dir data/inpainting-image-mask/imagenet/gt --save_root RESULT/imagenet_seg/gt --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_imagenet_seg_sketch_dual_encoder_res256_e299/imagenet_e_10_60_top1_nTpi-1_numSample1 --save_root RESULT/imagenet_seg/e --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_imagenet_seg_sketch_dual_encoder_res256_e299/imagenet_s_10_60_top1_nTpi-1_numSample1 --save_root RESULT/imagenet_seg/s --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_imagenet_seg_sketch_dual_encoder_res256_e299/imagenet_s_e_10_60_top1_nTpi-1_numSample1 --save_root RESULT/imagenet_seg/s_e --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_imagenet_seg_sketch_dual_encoder_res256_e299/imagenet_none_10_60_top1_nTpi-1_numSample1 --save_root RESULT/imagenet_seg/none --gpu 0
# python scripts/inference_seg.py --image_dir RESULT/tpami2024_vit_base_imagenet_seg_sketch_dual_encoder_res256_e299/imagenet_10_60_top1_nTpi-1_numSample1 --save_root RESULT/imagenet_seg/train_no_cond --gpu 0