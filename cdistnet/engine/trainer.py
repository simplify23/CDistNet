import torch
import math
import time
import datetime
from cdistnet.optim.loss import cal_performance
import os
# os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"

def train(model,
          train_data_loader,
          val_data_loader,
          optimizer,
          device,
          epoch,
          logger,
          meter,
          save_iter,
          display_iter,
          tfboard_iter,
          eval_iter,
          model_dir,
          label_smoothing,
          grads_clip,
          cfg,
          best_eval,
          best_epoch,
          best_iteration):
    model.train()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    max_iter = len(train_data_loader)
    end = time.time()
    total_time = 0.
    count = 0
    for iteration, batch in enumerate(train_data_loader, 0):
        if not batch:
            print('Error')
            continue
        meter.update_iter(max_iter * epoch + iteration)

        if cfg.train_method=='dist':
            images = batch[0].cuda(device,non_blocking=True)
            tgt = batch[1].cuda(device,non_blocking=True)
        else:
            images = batch[0].to(device)
            tgt = batch[1].to(device)

        optimizer.zero_grad()
        pred = model(images, tgt)
        #pred(b*tgt_len,vacab_size)

        tgt = tgt[:, 1:]
        # tgt(b,max_len)
        loss, n_correct = cal_performance(pred, tgt, smoothing=label_smoothing,local_rank=device)

        # torch.distributed.barrier()
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grads_clip)

        # optimizer.step()
        optimizer.step_and_update_lr(epoch)

        total_loss += loss.item()
        non_pad_mask = tgt.ne(0)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

        batch_time = time.time() - end
        end = time.time()
        total_time += batch_time
        count += 1
        avg_time = total_time / count
        eta_seconds = avg_time * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        acc = n_word_correct / n_word_total

        if tfboard_iter and (max_iter * epoch + iteration) % tfboard_iter == 0:
            meter.add_scalar(lr=optimizer._optimizer.param_groups[0]['lr'], loss=loss.item(), acc=acc)

        # if (max_iter * epoch + iteration) % 5000 == 0:
        #     # meter.add_graph(model.module,(images,tgt))
        #     for key in model.module.state_dict():
        #         meter.add_histogram(key,model.module.state_dict()[key])
        if cfg.train_method!='dist' or device ==0:
            if iteration % display_iter == 0:
                msg = 'epoch: {epoch}  iter: {iter}  loss: {loss: .6f}  lr: {lr: .6f}  eta: {eta}'.format(
                        epoch=epoch,
                        iter='{}/{}'.format(iteration, max_iter),
                        loss=loss.item(),
                        lr=optimizer._optimizer.param_groups[0]['lr'],
                        # lr=optimizer.param_groups[0]['lr'],
                        eta=eta_string
                    )
                logger.info(msg)

        # if save_iter and (max_iter * epoch + iteration) % save_iter == 0:
        #     logger.info("Saving model ...")
        #     torch.save(model.module.state_dict(), '{}/model_epoch_{}_iter_{}.pth'.format(model_dir, epoch, iteration))
        #     logger.info("Saved!")

        if epoch >= 6 and iteration % eval_iter == 0:
            eval_loss, eval_acc = eval(
                model=model,
                data_loader=val_data_loader,
                device=device,
                label_smoothing=label_smoothing,
                cfg=cfg
            )
            meter.add_scalar(eval_loss=eval_loss, eval_acc=eval_acc)
            logger.info('eval_loss:{:.4f},eval_acc:{:.4f}--------\n'.format(eval_loss,eval_acc))
            if eval_acc > best_eval:
                best_eval = eval_acc
                best_epoch = epoch
                best_iteration = iteration
                logger.info("Saving model: best_acc in epoch:{},iteration:{}".format(best_epoch,best_iteration))
                torch.save(model.module.state_dict(), '{}/epoch{}_best_acc.pth'.format(model_dir, epoch))
                logger.info("Saved!")
            if epoch > 8:
                logger.info("Saving last epoch model in epoch:{},iteration:{}".format(epoch, iteration))
                torch.save(model.module.state_dict(), '{}/epoch{}_iter{}.pth'.format(model_dir, epoch,iteration))
                logger.info("Saved!")
            model.train()

    loss_per_word = total_loss / max_iter
    accuracy = n_word_correct / n_word_total
    logger.info("Now: best_acc in epoch:{},iteration:{}".format(best_epoch, best_iteration))
    return loss_per_word, accuracy


def eval(model, data_loader, device, label_smoothing,cfg):
    model.eval()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    avg_acc = .0
    datasets_len = len(data_loader)
    with torch.no_grad():
        for dataset in data_loader:
            data_len=len(dataset)
            for iteration, batch in enumerate(dataset, 0):
                if cfg.train_method=='dist':
                    images = batch[0].cuda(device,non_blocking=True)
                    tgt = batch[1].cuda(device,non_blocking=True)
                else:
                    images = batch[0].to(device)
                    tgt = batch[1].to(device)
                pred = model(images, tgt)
                tgt = tgt[:, 1:]
                loss, n_correct = cal_performance(pred, tgt, smoothing=label_smoothing,local_rank=device)

                total_loss += loss.item()
                non_pad_mask = tgt.ne(0)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct

            loss_per_word = total_loss / data_len
            accuracy = n_word_correct / n_word_total
            # print("accuracy:{}".format(accuracy))
            avg_acc +=accuracy

    return loss_per_word, avg_acc/datasets_len


def do_train(model,
             train_dataloader,
             val_dataloader,
             optimizer,
             device,
             num_epochs,
             current_epoch,
             logger,
             meter,
             save_iter,
             display_iter,
             tfboard_iter,
             eval_iter,
             model_dir,
             label_smoothing,
             grads_clip,cfg):
    # meter.add_graph(model.module,(images,tgt))
    best_eval = 0.
    best_epoch = 0
    best_iteration = 0
    for epoch in range(current_epoch, num_epochs):
        if cfg.train_method=='dist':
            train_dataloader.sampler.set_epoch(epoch)
            val_dataloader.sampler.set_epoch(epoch)
        start = time.time()
        train_loss, train_accu = train(
            model=model,
            train_data_loader=train_dataloader,
            val_data_loader=val_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            meter=meter,
            save_iter=save_iter,
            display_iter=display_iter,
            tfboard_iter=tfboard_iter,
            eval_iter=eval_iter,
            model_dir=model_dir,
            label_smoothing=label_smoothing,
            grads_clip=grads_clip,
            cfg=cfg,
            best_eval=best_eval,
            best_epoch=best_epoch,
            best_iteration=best_iteration
        )

        logger.info('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, time: {time:3.3f} min'
                    .format(loss=train_loss, accu=100 * train_accu, time=(time.time() - start) / 60))

        # eval & save
        start = time.time()
        if epoch >= 6:
            logger.info("Start eval ...")
            val_loss, val_accu = eval(
                model=model,
                data_loader=val_dataloader,
                device=device,
                label_smoothing=label_smoothing,
                cfg=cfg,
            )
            logger.info('  - (Validation)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, time: {time:3.3f} min'
                        .format(loss=val_loss, accu=100 * val_accu, time=(time.time() - start) / 60))

            if cfg.train_method != 'dist' or device == 0:
                logger.info("Saving model ...")
                torch.save(model.module.state_dict(), '{}/model_epoch_{}.pth'.format(model_dir, epoch))
                logger.info("Saved!")