
#####################################
# experiments for CVPR2022
#####################################

# training commands for ffhq
# batch size 16, 8 GPUs in 1 node
python train_net.py --name cvpr2022_p_vqvae_ffhq --config_file configs/put_cvpr2022/ffhq/p_vqvae_ffhq.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 16
# batch size 6, 8 GPUs in 1 node
python train_net.py --name cvpr2022_transformer_ffhq --config_file configs/put_cvpr2022/ffhq/transformer_ffhq.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 6

# training commands for naturalscene
# batch size 16, 8 GPUs in 1 node
python train_net.py --name cvpr2022_p_vqvae_naturalscene --config_file configs/put_cvpr2022/naturalscene/p_vqvae_naturalscene.yaml --num_node 1 --tensorboard --auto_resume dataloader.batch_size 16
# batch size 6, 16 GPUs in 2 nodes
python train_net.py --name cvpr2022_transformer_naturalscene --config_file configs/put_cvpr2022/naturalscene/transformer_naturalscene.yaml --num_node 2 --tensorboard --auto_resume dataloader.batch_size 6

# training commands for imagenet
# batch size 16, 16 GPUs in 2 node
python train_net.py --name cvpr2022_p_vqvae_imagenet --config_file configs/put_cvpr2022/imagenet/p_vqvae_imagenet.yaml --num_node 2 --tensorboard --auto_resume dataloader.batch_size 16
# batch size 3, 32 GPUs in 4 nodes
python train_net.py --name cvpr2022_transformer_imagenet --config_file configs/put_cvpr2022/imagenet/transformer_imagenet.yaml --num_node 4 --tensorboard --auto_resume dataloader.batch_size 3


