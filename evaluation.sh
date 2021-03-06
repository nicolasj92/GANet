CUDA_VISIBLE_DEVICES=0 python evaluation.py --crop_height=480 \
                                            --crop_width=960 \
                                            --max_disp=192 \
                                            --data_path='./data/rvc_devkit/stereo/datasets_middlebury2014/eval/' \
                                            --test_list='./data/rvc_devkit/stereo/datasets_middlebury2014/lists/eval.list' \
                                            --save_path='./result/' \
                                            --kitti2015=1 \
                                            --threshold=3.0 \
                                            # --resume='./checkpoints/rvc.pth'
                                            --resume='./checkpoints/sceneflow_epoch_10.pth'
exit

CUDA_VISIBLE_DEVICES=0 python evaluation.py --crop_height=384 \
                                            --crop_width=1248 \
                                            --max_disp=192 \
                                            --data_path='/media/nicolas/Data_1/projects/rvc_devkit/stereo/datasets_middlebury2014/training/' \
                                            --test_list='/media/nicolas/Data_1/projects/rvc_devkit/stereo/datasets_middlebury2014/lists/training_kitti.list' \
                                            --save_path='./result/' \
                                            --kitti2015=1 \
                                            --threshold=3.0 \
                                            --resume='./checkpoints/kitti2015_final.pth'

# 2>&1 |tee logs/log_evaluation.txt
exit
CUDA_VISIBLE_DEVICES=0 python evaluation.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/ssd1/zhangfeihu/data/kitti2012/training/' \
                  --test_list='lists/kitti2012_train.list' \
                  --save_path='./result/' \
                  --resume='./checkpoint/kitti_final.pth' \
                  --threshold=3.0 \
                  --kitti=1
# 2>&1 |tee logs/log_evaluation.txt
exit
CUDA_VISIBLE_DEVICES=0 python evaluation.py --crop_height=576 \
                  --crop_width=960 \
                  --max_disp=192 \
                  --data_path='/ssd1/zhangfeihu/data/sceneflow/' \
                  --test_list='lists/sceneflow_test.list' \
                  --save_path='./result/' \
                  --resume='./checkpoint/sceneflow_epoch_10.pth' \
                  --threshold=1.0 
# 2>&1 |tee logs/log_evaluation.txt
exit

