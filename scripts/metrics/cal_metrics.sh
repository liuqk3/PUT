gt_dir=$1
result_dir=$2
device=$3
echo "gt_dir: ${gt_dir}"
echo "result_dir: ${result_dir}"
if [ ${#device} -eq 0 ]; then # not given the device, default is cuda:0
    device=cuda:0
elif [ $device -ge 0 ]; then # the given is such as 0, add prefix cuda:
    device=cuda:${device}
elif  [ ${#device} -eq 6 ]; then # the given is such as cuda:0, do nothing
    device=${device}
fi
echo "device: ${device}"

python scripts/metrics/cal_psnr.py --gt_dir ${gt_dir} --result_dir ${result_dir}
python scripts/metrics/cal_fid.py --path1 ${gt_dir} --path2 ${result_dir} --device ${device} --net inception
python scripts/metrics/cal_lpips.py --type perceptron --path1 ${gt_dir} --path2 ${result_dir} --device ${device}
