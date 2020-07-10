
# KITTI Dimension Settings
python predict.py --crop_height=336 \
                  --crop_width=1200 \
                  --max_disp=192 \
                  --data_path='/media/nicolas/projects/rvc_devkit/stereo/datasets_middlebury2014/test/' \
                  --test_list='/media/nicolas/projects/rvc_devkit/stereo/datasets_middlebury2014/lists/test.list' \
                  --save_path='./result/' \
                  --kitti2015=1 \
                  --resume='./checkpoints/kitti2015_final.pth' \
                  --name='GANet_RVC'
exit

python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/media/feihu/Storage/stereo/kitti/testing/' \
                  --test_list='lists/kitti2012_test.list' \
                  --save_path='./result/' \
                  --kitti=1 \
                  --resume='./checkpoint/kitti2012_final.pth'



