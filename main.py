#!/usr/bin/env python3

import torch
import Dataset
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import Nets
from tqdm import tqdm
import torch.optim as optim
from Eveluate import metrics
import Framework

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser(description='Super Resolution and Semantic Segmentation')
    parser.add_argument("--dataset", type=str, default="MultiData_wTC")
    parser.add_argument("--hr_batch_size", type=int, default=32)
    parser.add_argument("--lr_batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--sr", type=str,
                        default="EDSR")
    parser.add_argument("--ss", type=str,
                        default="Unet")
    parser.add_argument("--framework", type=str,
                        default="lr_lr_and_lr_hr_wTC")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--valid_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00015)
    parser.add_argument("--save_path", type=str, default="./model.ckpt")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()
    args.save_path = "./" + args.dataset + args.sr + args.ss + "-model"
    return args


# In order to reconstruct the results
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def valid(dataloader, valid_framework, evaluator):
    valid_framework.eval()
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid Mode", unit=" uttr")
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, out, label = valid_framework(batch, "val")
            for v in loss.values():
                total_loss += v.sum()
            out = torch.sigmoid(out)
            out = out.detach().cpu().numpy()
            out = np.where(out > 0.5, 1, 0)
            label = label.detach().cpu().numpy()
            evaluator.add_batch(label, out)
        pbar.update(dataloader.batch_size)
    IoU = evaluator.IoU()
    OA = evaluator.OA()
    F1 = evaluator.F1()
    pbar.set_postfix(
        loss=f"{total_loss :.4f}",
        IoU=f"{IoU :.4f}",
        OA=f"{OA :.4f}",
        F1=f"{F1 :.4f}",
    )
    evaluator.reset()
    pbar.close()
    valid_framework.train()
    return total_loss / len(dataloader), IoU, OA, F1


def main_epoch(args):
    LR_HRLoader, LR_LRLoader, ValLoader = Dataset.setup_loaders(args)
    framework = Framework.setup_framework(args)
    framework = framework.to(device)

    framework = Nets.wrap_network_in_dataparallel(framework, False)
    if args.resume:
        framework.load_state_dict(torch.load(args.resume))
    print(f"{args.framework} build successful! sr is {args.sr}, and ss is {args.ss} ")
    total_epochs = args.epochs
    total_steps = len(LR_LRLoader)
    pbar = tqdm(total=total_epochs, desc="Train Mode", unit="epoch")
    optimizer = optim.AdamW(framework.parameters(), lr=args.lr)
    # optimizer = optim.SGD([{'params': sr.parameters()}, {'params': ss.parameters()}], lr=args.lr, momentum=0.9,
    #                       weight_decay=0.0001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6, last_epoch=-1)
    print(f"optimizer is AdmW, learning rate is {args.lr}")
    evaluator = metrics.Evaluator(num_class=2)  # 前景和背景两类
    writer = SummaryWriter(
        comment=args.dataset + args.sr + args.ss + "_" + args.framework)  # save at run/日期时间-args.networks
    best_accuracy = -1
    """
    Training phases
    """

    for epoch in range(total_epochs):
        batch_loss = 0.0
        hr_loss = 0.0
        lr_loss = 0.0
        lr_ce_loss = 0.0
        pbar.update()
        # to avoid memory leaks
        LR_HRLoader_iter = iter(LR_HRLoader)
        for i, LR_LRbatch in enumerate(LR_LRLoader):
            try:
                LR_HRbatch = next(LR_HRLoader_iter)
            except StopIteration:
                LR_HRLoader_iter = iter(LR_HRLoader)
                LR_HRbatch = next(LR_HRLoader_iter)

            loss1, hr_pre = framework(LR_HRbatch, "train_hr")
            loss2 = framework(LR_LRbatch, "train_lr")
            total_loss = 0.0
            for v in loss1.values():
                total_loss += v
            for v2 in loss2.values():
                total_loss += v2

            hr_gt = LR_HRbatch["label"].detach().cpu().numpy()
            hr_pre = torch.sigmoid(hr_pre)
            hr_pre = hr_pre.detach().cpu().numpy()
            hr_pre = np.where(hr_pre > 0.5, 1, 0) # the threshold for binarization is 0.5
            evaluator.add_batch(hr_gt, hr_pre)

            total_loss = total_loss.sum()
            batch_loss += total_loss.detach().cpu().item()

            hr_loss += loss1["hr_loss"].sum().detach().cpu().item()
            lr_ce_loss += loss2["lr_bce_loss"].sum().detach().cpu().item()
            lr_loss += loss2["lr_sr_loss"].sum().detach().cpu().item()

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # pbar.update()
            pbar.set_postfix(
                batch=f"{i}/{total_steps}",
                loss=f"{total_loss.detach().cpu().item():.4f}",
                epoch=epoch + 1  # ,
                # lr = scheduler.get_lr()[0]
            )
        IoU = evaluator.IoU()
        pbar.set_postfix(
            epoch_sum_loss=f"{batch_loss:.4f}",
            IoU=f"{IoU:.4f}",
            epoch=epoch + 1
        )

        writer.add_scalar(tag="train loss", scalar_value=batch_loss, global_step=epoch + 1)
        writer.add_scalar(tag="train IoU", scalar_value=IoU, global_step=epoch + 1)
        writer.add_scalar(tag="hr loss", scalar_value=hr_loss, global_step=epoch + 1)
        writer.add_scalar(tag="lr loss", scalar_value=lr_loss, global_step=epoch + 1)
        writer.add_scalar(tag="lr ce loss", scalar_value=lr_ce_loss, global_step=epoch + 1)
        evaluator.reset()
        """
        Validation phases
        """
        if (epoch + 1) % args.valid_epochs == 0:
            pbar.close()
            val_loss, val_IoU, val_OA, val_F1 = valid(ValLoader, framework, evaluator)
            writer.add_scalar("val loss", scalar_value=val_loss, global_step=epoch + 1)
            writer.add_scalar("val IoU", scalar_value=val_IoU, global_step=epoch + 1)
            writer.add_scalar("val OA", scalar_value=val_OA, global_step=epoch + 1)
            writer.add_scalar("val F1", scalar_value=val_F1, global_step=epoch + 1)
            # keep the best model, using IoU as metrics
            if val_IoU > best_accuracy:
                best_accuracy = val_IoU
                best_state_dict = framework.state_dict()

            pbar = tqdm(total=total_epochs, desc="Train Mode", unit="step", initial=epoch + 1)  # 重启 pbar

        if (epoch + 1) % args.save_epochs == 0:
            state_dict = framework.state_dict()
            torch.save(state_dict, args.save_path + f"-{str(epoch + 1)}.ckpt")

        if (epoch + 1) % args.valid_epochs == 0 and best_state_dict is not None:
            torch.save(best_state_dict, args.save_path + "-best.ckpt")
            pbar.write(f"epoch {epoch + 1}, best model saved. (accuracy={best_accuracy:.4f})")
            best_state_dict = None
        # scheduler.step()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = get_args()
    print(args)
    # seed = 300
    # same_seeds(seed)
    # print("random seed is ", str(seed))
    main_epoch(args)
