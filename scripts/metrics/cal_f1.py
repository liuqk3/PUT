import os
from os.path import join
import torch
from torchmetrics.functional import kl_divergence
from PIL import Image
import numpy as np

def cal_f1_score(args):
    precision = []
    recall = []
    f1 = []
    imgpaths = os.listdir(args.gt_dir)
    for imgpath in imgpaths:
        gt_img = join(args.gt_dir, imgpath)
        img = join(args.result_dir, imgpath)

        gt_img = np.array(Image.open(gt_img))
        img = np.array(Image.open(img))
        # import pdb; pdb.set_trace()
        gt_img = torch.tensor(gt_img).reshape(1,-1)
        gt_img = (gt_img < 100).float()
        img = torch.tensor(img).reshape(1,-1)
        img = (img < 100).float()

        pre_ans = torch.mean(gt_img * img) / torch.mean(img)
        recall_ans = torch.mean(gt_img * img) / torch.mean(gt_img)
        f1_ans = 2 * pre_ans * recall_ans / (pre_ans + recall_ans)

        f1.append(f1_ans)
        precision.append(pre_ans)
        recall.append(recall_ans)

    # print(result, ': ')
    # print("Recall: {}".format(np.mean(recall)))
    # print("Precision: {}".format(np.mean(precision)))
    print("F1: {}".format(np.mean(f1)))


def get_args():
    parser = argparse.ArgumentParser(description='Calculate the F1 score between two edge maps.')
    parser.add_argument('--gt_dir', type=str, default='path/to/sketch_map/of/gt_images',
                        help='sketch map of gt images need to be loaded') 
    parser.add_argument('--result_dir', type=str, default='path/to/sketch_map/of/inpainted_images',
                        help='sketch map of inpainted images need to be loaded') 
    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    return args


if __name__ == '__main__':

    args = get_args()
    cal_f1_score(args)