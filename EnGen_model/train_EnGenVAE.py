import os
import sys
import time
import torch
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from sklearn import preprocessing
from models import EnGenVAE

from utils_EnGenVAE import cytofDataset, GlobalsVars




def train_engen_vae(args, globals_vars):

    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda:{}'.format(args.GPU_ID) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    cytof_dataset = cytofDataset(globals_vars=globals_vars)
    dataloader = DataLoader(cytof_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.n_cpus, pin_memory=True)

    def loss_fn(recon_x, x, mean, log_var):
        # BCE = torch.nn.functional.binary_cross_entropy(
        #     recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        reconstruction_loss = nn.MSELoss(reduction='sum')
        MSE = reconstruction_loss(recon_x, x)
        # print('KLD = {}'.format(KLD))
        # print('MSE = {}'.format(MSE))
        # print('x.size(0) = {}'.format(x.size(0)))
        return (MSE + (args.lambda_)*KLD) / x.size(0), MSE + (args.lambda_)*KLD, x.size(0), MSE, (args.lambda_)*KLD

    engen_vae = EnGenVAE(
        encoder_layer_sizes=args.encoder_layer_sizes, 
        latent_size=args.latent_size, 
        decoder_layer_sizes=args.decoder_layer_sizes,
        device=device).to(device)

    optimizer = torch.optim.Adam(engen_vae.parameters(), lr=args.learning_rate)
    
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.sched_gamma, patience=args.sched_patience, cooldown=10, min_lr=10e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.sched_gamma, patience=args.sched_patience, cooldown=10, min_lr=10e-5)

    logs = defaultdict(list)

    # Load checkpoint if available
    if args.ckpt is not None:
        print('Loading last checkpoint..')
        engen_vae.load_ckpt(args.ckpt)

    engen_vae.to(device)

    Tensor = torch.cuda.FloatTensor if device=='cuda:{}'.format(args.GPU_ID) else torch.FloatTensor

    best_model_dict = {'epoch':0, 'min_total_loss':float('inf')} 
    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))
        engen_vae.train()
        total_loss = 0
        total_loss_raw = 0
        total_recon_loss = 0
        total_kl_loss = 0
        pbar = tqdm(dataloader, ascii=True, desc="Epoch: {:2d}/{}".format(epoch + 1, args.epochs))
        current_lr = optimizer.param_groups[0]['lr']
    
        for data in pbar:

            x_in = Variable(data['row_source'].type(Tensor)).to(device)
            x_out = Variable(data['row_target'].type(Tensor)).to(device)

            recon_x, mean, log_var, z = engen_vae(x_in)
    
            loss, loss_raw, x_size, MSE_loss, KLD_loss = loss_fn(recon_x,x_out, mean, log_var)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Bookkeeping
            if total_loss == 0:
                total_loss = loss.item()
                total_recon_loss = MSE_loss.item()
                total_kl_loss = KLD_loss.item()
            else:
                total_loss = total_loss * .95 + loss.item() * .05
                total_loss_raw = total_loss_raw * .95 + loss_raw.item() * .05
                total_recon_loss = total_recon_loss * .95 + MSE_loss.item() * .05
                total_kl_loss = total_kl_loss * .95 + KLD_loss.item() * .05
                           
            pbar.set_postfix({'loss':loss.item(), 'loss_raw':total_loss_raw, 'lr':current_lr,
                'MSE_loss':total_recon_loss, 'KLD_loss':total_kl_loss})
       
        scheduler.step(loss) 

        logs['epoch'].append(epoch)
        logs['loss'].append(loss.item())
        logs['loss_raw'].append(total_loss_raw)
        logs['learning_rate'].append(current_lr)
        logs['MSE_loss'].append(total_recon_loss)
        logs['KLD_loss'].append(total_kl_loss)

 
        if epoch % args.print_every == 0 or epoch == args.epochs - 1 or (total_loss < best_model_dict['min_total_loss'] and epoch >= best_model_dict['epoch']+20):
            engen_vae.eval()

            filename = globals_vars.dir_path_csv+'epoch{}_model_params.txt'.format(epoch) 
            with open(filename, 'w') as filetowrite:
                model_params = [str(param) for param in engen_vae.parameters()]
                filetowrite.writelines(model_params)
                filetowrite.close()
            
            if total_loss < best_model_dict['min_total_loss'] and epoch >= best_model_dict['epoch']+20:
                print('********* best model found so far: total_loss={}'.format(total_loss))
                engen_vae.save_ckpt(globals_vars.dir_path_ckpt+'best_model_engen_vae.pth')
                with open(globals_vars.dir_path_ckpt+'best_model_engen_vae.txt', 'a+') as f:
                    f.write('\n===========================================\n')
                    f.write('\n'.join("{}={}".format(key,val[-1]) for (key,val) in logs.items()))
                best_model_dict['epoch'] = epoch
                best_model_dict['min_total_loss'] = total_loss
            else:
                engen_vae.save_ckpt(globals_vars.dir_path_ckpt+'last_model_engen.pth')
            if epoch % 500 == 499:
                engen_vae.save_ckpt(globals_vars.dir_path_ckpt+'epoch_{}_model_engen_vae.pth'.format(epoch))
           
            pd_logs = pd.DataFrame(logs)
            pd_logs.to_csv(globals_vars.dir_path_csv+'Logs_engen_vae.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_id", type=int, required=True, help='iterate int from 0 to n_iterations-1') #repeat for 30 iterations
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--GPU_ID", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_cpus", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    # parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[37, 128, 256, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 256, 128, 37])
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--lambda_", type=float, default=0.01)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--csv_root", type=str, default='csv')
    parser.add_argument('--ckpt', type=str, default=None, metavar='PATH',
    # parser.add_argument('--ckpt', type=str, default='PATH_TO_SAVED_CHECKPOINT', metavar='PATH',
                        help='checkpoint path to load from/save to model (default: None)')
    parser.add_argument('--sched_step', default=30, type=int, help='scheduler steps for rate update')
    parser.add_argument('--sched_gamma', default=0.5, type=float, help='scheduler gamma for rate update')
    parser.add_argument('--sched_patience', default=20, type=float,
                        help='scheduler patience for rate update - pretrain')
    parser.add_argument("--about", type=str, default='x_in: {}, x_out: {}, matched'.format('Pre', 'Post'))
    
    args = parser.parse_args()
    globals_vars = GlobalsVars(args.iter_id)
    if args.ckpt is not None: # continue from the checkpoint
        assert os.path.exists(args.ckpt), "Saved model not found!"
    
    text_filename = globals_vars.dir_path_main+'/commandline_args.txt'
    with open(text_filename, 'w') as f:
        f.write('\n'.join("{}={}".format(key,val) for (key,val) in vars(args).items()))
    train_engen_vae(args, globals_vars)
 