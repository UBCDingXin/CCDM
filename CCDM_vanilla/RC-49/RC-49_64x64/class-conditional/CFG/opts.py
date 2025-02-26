import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--eval_ckpt_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--setting_name', type=str, default='Setup1')

    ''' Dataset '''
    parser.add_argument('--data_split', type=str, default='train', choices=['all', 'train'])
    parser.add_argument('--min_label', type=float, default=0)
    parser.add_argument('--max_label', type=float, default=90.0)
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=64, metavar='N')
    parser.add_argument('--max_num_img_per_label', type=int, default=25, metavar='N')
    parser.add_argument('--max_num_img_per_label_after_replica', type=int, default=0, metavar='N')
    parser.add_argument('--num_classes', type=int, default=150, required=True)

    ''' Training '''
    parser.add_argument('--niters', type=int, default=100000, metavar='N')
    parser.add_argument('--resume_niter', type=int, default=0, metavar='N')
    parser.add_argument('--timesteps', type=int, default=1000, metavar='N')
    parser.add_argument('--sampling_timesteps', type=int, default=250, metavar='N')
    parser.add_argument('--train_batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--train_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--train_amp', action='store_true', default=False)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1, metavar='N')
    parser.add_argument('--sample_every', type=int, default=1000, metavar='N')
    parser.add_argument('--save_every', type=int, default=10000, metavar='N')

    ''' Sampling and Evaluation '''
    parser.add_argument('--comp_FID', action='store_true', default=False)
    parser.add_argument('--dump_fake_data', action='store_true', default=False)
    parser.add_argument('--num_eval_labels', type=int, default=-1)
    parser.add_argument('--samp_batch_size', type=int, default=200)
    parser.add_argument('--nfake_per_label', type=int, default=200)
    parser.add_argument('--epoch_FID_CNN', type=int, default=200)
    parser.add_argument('--FID_radius', type=int, default=0)
    parser.add_argument('--FID_num_centers', type=int, default=-1)
    parser.add_argument('--dump_fake_for_NIQE', action='store_true', default=False)
    parser.add_argument('--comp_IS_and_FID_only', action='store_true', default=False)
    parser.add_argument('--niqe_dump_path', type=str, default='None')
    parser.add_argument('--eval_batch_size', type=int, default=200, required=False)

    args = parser.parse_args()

    return args