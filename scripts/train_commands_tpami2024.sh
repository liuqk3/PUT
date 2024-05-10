
################################################
# experiments for TPAMI2024, resolution 256x256
################################################

# training commands for ffhq
# batch size 24, 8 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_ffhq_res256 --config_file configs/put_tpami2024/ffhq/p_vqvae_ffhq_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 24
# batch size 6, 9 GPUs in 1 node
python train_net.py --name tpami2024_vit_base_ffhq_res256 --config_file configs/put_tpami2024/ffhq/vit_base_ffhq_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 6

# training commands for naturalscene
# batch size 28, 8 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_naturalscene_res256 --config_file configs/put_tpami2024/naturalscene/p_vqvae_naturalscene_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 28
# batch size 12, 16 GPUs in 2 nodes
python train_net.py --name tpami2024_transformer_naturalscene_res256 --config_file configs/put_tpami2024/naturalscene/vit_base_naturalscene_res256.yaml --num_node 2 --tensorboard --auto_resume dataloader.batch_size 12

# training commands for imagenet
# batch size 28, 9 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_imagenet_res256 --config_file configs/put_tpami2024/imagenet/p_vqvae_imagenet_res256.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 28
# batch size 12, 32 GPUs in 4 nodes
python train_net.py --name tpami2024_transformer_imagenet --config_file configs/put_tpami2024/imagenet/vit_base_imagenet_res256.yaml --num_node 4 --tensorboard --auto_resume dataloader.batch_size 12



################################################
# experiments for TPAMI2024, resolution 512x512
################################################

# training commands for ffhq
# batch size 8, 9 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_ffhq_res512 --config_file configs/put_tpami2024/ffhq/p_vqvae_ffhq_res512.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 8
# batch size 6, 9 GPUs in 1 node
python train_net.py --name tpami2024_vit_base_ffhq_res512 --config_file configs/put_tpami2024/ffhq/vit_base_ffhq_res512.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 6

# training commands for naturalscene
# batch size 24, 8 GPUs in 1 node
python train_net.py --name tpami2024_p_vqvae_naturalscene_res512 --config_file configs/put_tpami2024/naturalscene/p_vqvae_naturalscene_res512.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 24
# batch size 12, 16 GPUs in 2 nodes
python train_net.py --name tpami2024_transformer_naturalscene_res256 --config_file configs/put_tpami2024/naturalscene/vit_base_naturalscene_res512.yaml --num_node 2 --tensorboard --auto_resume dataloader.batch_size 12

# training commands for imagenet
# batch size 24, 16 GPUs in 2 node
python train_net.py --name tpami2024_p_vqvae_imagenet --config_file configs/put_tpami2024/imagenet/p_vqvae_imagenet_res512.yaml --num_node 2 --tensorboard --auto_resume dataloader.batch_size 24
# batch size 12, 32 GPUs in 4 nodes
python train_net.py --name tpami2024_transformer_imagenet --config_file configs/put_tpami2024/imagenet/vit_base_imagenet_res512.yaml --num_node 4 --tensorboard --auto_resume dataloader.batch_size 12

