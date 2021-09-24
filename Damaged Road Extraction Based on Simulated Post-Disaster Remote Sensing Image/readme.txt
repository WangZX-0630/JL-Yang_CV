Before you run the program, install the modules first by command:
pip install -r requirements.txt

Then make sure the following paths have no file:
imgs/ade20k/validation
imgs/ade20k/erased
submits/label
submits/imgs
submits/eval
submits/compare
dataset/post_disaster
dataset/pre_disaster

Also make sure the following paths have been deleted:
submits/post_disaster
submits/pre_disaster

Then run CoCosNet_test.py by command:
python CoCosnet_test.py --name ade20k --dataset_mode ade20k --dataroot ./imgs/ade20k --gpu_ids 0 --nThreads 0 --batchSize 6 --use_attention --maskmix --warp_mask_losstype direct --PONO --PONO_C  --save_per_img

User should store the original remote sensing images and their semantic segmentation masks in imgs/ade20k/validation and keep the file names in pair
like a.jpg for original remote sensing images and a.png for its semantic segmentation mask

The results are stored in the following path:
altered mask: imgs/ade20k/validation
erased mask: imgs/ade20k/erased
simulated post-disaster images: output/test_per_img/ade20k
semantic segmentation masks of pre_disaster images: submits/pre_disaster
semantic segmentation masks of post_disaster images: submits/post_disaster
initial semantic segmentation masks of damaged road: submits/compare
denoised semantic segmentation masks of damaged road: submits/img
groundtruth of damaged road: submits/label
evaluation of damaged road extraction: submits/eval
