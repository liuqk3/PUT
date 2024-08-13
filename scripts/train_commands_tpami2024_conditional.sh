
################################################
# experiments for TPAMI2024 conditional
################################################

# training commands for ffhq
# segmentation encoder: batch size 24, 8 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_ffhq_seg_res256 --config_file configs/put_tpami2024_conditional/ffhq/p_vqvae_ffhq_seg_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 24
# sketch encoder: batch size 24, 8 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_ffhq_sketch_res256 --config_file configs/put_tpami2024_conditional/ffhq/p_vqvae_ffhq_sketch_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 24
# transformer: batch size 12, 16 GPUs in 2 node
python train_net.py --name tpami2024_vit_base_ffhq_seg_sketch_dual_encoder_res256 --config_file configs/put_tpami2024_conditional/ffhq/vit_base_ffhq_seg_sketch_res256.yaml --num_node 2 --tensorboard --auto_resume dataloader.batch_size 12

# training commands for naturalscene
# segmentation encoder: batch size 28, 8 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_naturalscene_seg_res256 --config_file configs/put_tpami2024_conditional/imagenet/p_vqvae_imagenet_seg_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 28
# sketch encoder: batch size 28, 8 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_naturalscene_sketch_res256 --config_file configs/put_tpami2024_conditional/imagenet/p_vqvae_imagenet_sketch_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 28
# transformer: batch size 12, 16 GPUs in 2 nodes
python train_net.py --name tpami2024_vit_base_naturalscene_seg_sketch_dual_encoder_res256 --config_file configs/put_tpami2024/naturalscene/vit_base_naturalscene_res256.yaml --num_node 2 --tensorboard --auto_resume dataloader.batch_size 12

# training commands for imagenet
# segmentation encoder: batch size 28, 9 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_imagenet_seg_res256 --config_file configs/put_tpami2024_conditional/imagenet/p_vqvae_imagenet_seg_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 28
# sketch encoder batch size 28, 9 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_imagenet_sketch_res256 --config_file configs/put_tpami2024_conditional/imagenet/p_vqvae_imagenet_sketch_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 28
# transformer: batch size 8, 48 GPUs in 6 nodes
python train_net.py --name tpami2024_vit_base_imagenet_seg_sketch_dual_encoder_res256 --config_file configs/put_tpami2024/imagenet/vit_base_imagenet_res256.yaml --num_node 6 --tensorboard --auto_resume dataloader.batch_size 8


