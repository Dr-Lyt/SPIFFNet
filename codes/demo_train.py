
from option import args
import data
import model
import utils
import loss
import trainer
import torch
import thop
from torchstat import stat


if __name__ == '__main__':

    args.save = 'SPIFFNet'
    args.model = 'spiffnet'
    # args.model = 'lgcnet'
    # args.model = 'srcnn'

    args.decay_type = 'cosine'
    args.lr_max = 4e-4
    args.lr_min = 1e-7
    args.lr_cos_decay = 2000

    args.scale = [4]
    args.resume = 0
    args.epochs = 2000
    args.print_model = True
    args.train_patch_size = 192
    args.test_patch_size = 256

    # args.swinir_depths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # args.swinir_num_heads = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    # args.swinir_window_size = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    args.swinir_depths = [1, 1, 1, 1, 1]
    args.swinir_num_heads = [4, 4, 4, 4, 4]
    args.swinir_window_size = [16, 16, 16, 16, 16]
    # args.swinir_num_heads = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # args.swinir_num_heads = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # args.swinir_num_heads = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    # args.swinir_num_heads = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]


    args.swinir_embed_dim = 64
    args.batch_size = 4

    args.test_block = True
    # args.test_only = True

    checkpoint = utils.checkpoint(args)
    if checkpoint.ok:
        dataloaders = data.create_dataloaders(args)
        sr_model = model.Model(args, checkpoint)
        sr_loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = trainer.Trainer(args, dataloaders, sr_model, sr_loss, checkpoint)

        print(sr_model.flops() / 1e9)

        while not t.terminate():
            t.train()
            t.test()

    checkpoint.done()
