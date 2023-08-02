
import argparse
import template

parser = argparse.ArgumentParser(description='Super-resolution')

parser.add_argument('--debug', action='store_true', default=False,
                    help='Enables debug mode')

# hardware specifications
parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# data specifications
parser.add_argument('--dataset', type=str, default='UCMerced',
                    help='train dataset name')
parser.add_argument('--dir_data', type=str, default='.',
                    help='dataset directory')
parser.add_argument('--data_test', type=str, default='.',
                    help='test dataset name')
parser.add_argument('--image_size', type=int, default=256,
                    help='train/test reference image size')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--patch_size', type=int, default=256,
                    help='output patch size for training')
parser.add_argument('--cubic_input', action='store_true', default=False,
                    help='LR images are firstly upsample by cubic interpolation')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension: '
                         'sep_reset - first convert img to .npy and read .npy; '
                         'sep - read .npy from disk; '
                         'img - read image from disk; '
                         'ram - load image into RAM memory')
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true', default=False,
                    help='enable memory-efficient forward')
parser.add_argument('--test_y', action='store_true', default=False,
                    help='test on Y channel')
parser.add_argument('--test_patch', action='store_true', default=False,
                    help='test on patches rather than the whole image')
parser.add_argument('--test_block', action='store_true', default=False,
                    help='test by blcok-by-block')

# model specifications
parser.add_argument('--model', default='TransENet',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--n_basic_modules', type=int, default=10,
                    help='number of basic modules')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# SwinIR setting
parser.add_argument('--swinir_image_size', type=int, default=64,
                    help='image_size of swinir')
parser.add_argument('--swinir_patch_size', type=int, default=1,
                    help='patch_size of swinir')
parser.add_argument('--swinir_window_size', type=int, default=8,
                    help='window_size of swinir')
parser.add_argument('--swinir_fre_stage', type=int, default=8,
                    help='frequency stage of swinir')
parser.add_argument('--swinir_kernal_size', type=int, default=21,
                    help='window_size of swinir')
parser.add_argument('--swinir_depths', type=list, default=[6, 6, 6, 6, 6, 6],
                    help='depths of swinir')
parser.add_argument('--swinir_embed_dim', type=int, default=96,
                    help='embed_dim of swinir')
parser.add_argument('--swinir_num_heads', type=list, default=[6, 6, 6, 6, 6, 6],
                    help='num_heads of swinir')
parser.add_argument('--swinir_mlp_ratio', type=int, default=2,
                    help='mlp_ratio of swinir')
parser.add_argument('--swinir_upsampler', type=str, default='pixelshuffle',
                    help='upsampler of swinir')
parser.add_argument('--swinir_resi_connection', type=str, default='1conv',
                    help='resi_connection of swinir')
parser.add_argument('--swinir_init_type', type=str, default='default',
                    help='init_type of swinir')
parser.add_argument('--swinir_lr', type=float, default=2e-4,
                    help='learning rate of swinir')


# training specifications
parser.add_argument('--reset', action='store_true', default=False,
                    help='reset the training')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true', default=False,
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true', default=False,
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--test_metric', type=str, default='psnr',
                    help='for best model selection in test phase (psnr, ssim)')

# optimization specifications
parser.add_argument('--w', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=750,
                    help='learning rate decay per N epochs')
parser.add_argument('--lr_min', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_max', type=float, default=3e-4,
                    help='learning rate')
parser.add_argument('--lr_cos_decay', type=int, default=1500,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.99,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# log specifications
parser.add_argument('--save', type=str, default='tes_1',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_false', default=True,
                    help='print model')
parser.add_argument('--save_models', action='store_true', default=False,
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', default=False,
                    help='save output results')

# Option for TransENet
parser.add_argument('--back_projection_iters', type=int, default=10,
                    help='back projection iterations')
parser.add_argument('--en_depth', type=int, default=8,
                    help='the depth of encoder')
parser.add_argument('--de_depth', type=int, default=1,
                    help='the depth of decoder')

args = parser.parse_args()
args.scale = list(map(lambda x: int(x), args.scale.split('+')))

template.set_template(args)