import os
import time
import socket
from datetime import datetime
import torch
from tensorboardX import SummaryWriter
from DORNnet import DORN
from load_data import getNYUDataset, get_depth_sid
import m_utils
from m_utils import ordLoss, update_ploy_lr, save_checkpoint
from error_metrics import AverageMeter, Result

init_lr = 0.0001
momentum = 0.9
epoches = 140
batch_size = 2
max_iter = 9000000
resume = True # 是否有已经保存的模型
model_path = '.\\run\\checkpoint-119.pth.tar' # 注意修改加载模型的路径
#model_path = '.\\run\\model_best.pth.tar' # 注意修改加载模型的路径
output_dir = '.\\run'


def main():
    train_loader, val_loader, test_loader = getNYUDataset()
    print("已经获取数据")
    # 先把结果设置成最坏
    best_result = Result()
    best_result.set_to_worst()

    if resume:
        # TODO
        # best result应当从保存的模型中读出来
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model_dict = checkpoint['model']
        # model = DORN()
        # model.load_state_dict(model_dict)
        model = checkpoint['model']
        # 使用SGD进行优化
        # in paper, aspp module's lr is 20 bigger than the other modules
        aspp_params = list(map(id, model.aspp_module.parameters()))
        base_params = filter(lambda p: id(p) not in aspp_params, model.parameters())
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': model.aspp_module.parameters(), 'lr': init_lr * 20},
        ], lr=init_lr, momentum=momentum)

        print("loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        del checkpoint  # 删除载入的模型
        del model_dict
        print("加载已经保存好的模型")

    else:
        print("创建模型")
        model = DORN()
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum)
        start_epoch = 0

    if torch.cuda.device_count():
        print("当前GPU数量：", torch.cuda.device_count())
        # model = torch.nn.DataParallel(model)

    model = model.cuda()
    # 定义损失函数
    criterion = ordLoss()
    # 初始化输出文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    best_txt = os.path.join(output_dir, 'best.txt')
    log_path = os.path.join(output_dir, 'logs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)



    # 开始训练
    for  epoch in range(start_epoch, epoches):
        train(train_loader, model, criterion, optimizer, epoch, logger)
        # 验证
        result, img_merge = validate(val_loader, model, epoch, logger)
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}\nrmse={:.3f}\nrml={:.3f}\nlog10={:.3f}\nd1={:.3f}\nd2={:.3f}\ndd31={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                               result.delta3,
                               result.gpu_time))
            if img_merge is not None:
                img_filename = output_dir + '/comparison_best.png'
                m_utils.save_image(img_merge, img_filename)

        # 每个epoch保存检查点
        save_checkpoint({'epoch': epoch, 'model': model, 'optimizer': optimizer, 'best_result': best_result},
                        is_best, epoch, output_dir)
        print("模型保存成功")


# 在NYU训练集上训练一个epoch
def train(train_loader, model, criterion, optimizer, epoch, logger):
    average_meter = AverageMeter()
    model.train()
    end = time.time()
    batch_num = len(train_loader)
    current_step = batch_num * batch_size * epoch
    for i, (input, target) in enumerate(train_loader):
        lr = update_ploy_lr(optimizer, init_lr, current_step, max_iter)

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
        data_time = time.time() - end

        current_step += input.data.shape[0]

        if current_step == max_iter:
            logger.close()
            print("迭代完成")
            break
        torch.cuda.synchronize()

        end = time.time()
        # compute pred
        end = time.time()
        with torch.autograd.detect_anomaly():
            pred_d, pred_ord = model(input)  # @wx 注意输出

            loss = criterion(pred_ord, target)
            optimizer.zero_grad()
            loss.backward()  # compute gradient and do SGD step
            optimizer.step()

        torch.cuda.synchronize()

        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        depth = get_depth_sid(pred_d)
        target_dp = get_depth_sid(target)
        result.evaluate(depth.data, target_dp.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % 10 == 0:
            print('Train Epoch: {0} [{1}/{2}]\t'
                  'learning_rate={lr:.8f} '
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={loss:.3f} '
                  'RMSE={result.rmse:.3f}({average.rmse:.3f}) '
                  'RML={result.absrel:.3f}({average.absrel:.3f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                epoch, i + 1, batch_num, lr=lr, data_time=data_time, loss=loss.item(),
                gpu_time=gpu_time, result=result, average=average_meter.average()))

            logger.add_scalar('Learning_rate', lr, current_step)
            logger.add_scalar('Train/Loss', loss.item(), current_step)
            logger.add_scalar('Train/RMSE', result.rmse, current_step)
            logger.add_scalar('Train/rml', result.absrel, current_step)
            logger.add_scalar('Train/Log10', result.lg10, current_step)
            logger.add_scalar('Train/Delta1', result.delta1, current_step)
            logger.add_scalar('Train/Delta2', result.delta2, current_step)
            logger.add_scalar('Train/Delta3', result.delta3, current_step)
        avg = average_meter.average()

def validate(val_loader, model, epoch, logger, write_to_file=True):
    average_meter = AverageMeter()
    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        # 计算数据时间
        data_time = time.time() - end
        with torch.no_grad():
            pred_d, pred_ord = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # 度量
        result = Result()
        depth = get_depth_sid(pred_d)
        target_dp = get_depth_sid(target)
        result.evaluate(depth.data, target_dp.data)

        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # 保存一些验证结果
        skip = 11
        rgb = input
        if i == 0:
            img_merge = m_utils.merge_into_row(rgb, target_dp, depth)
        elif (i < 8 * skip) and (i % skip == 0):
            row = m_utils.merge_into_row(rgb, target_dp, depth)
            img_merge = m_utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_dir + '/comparison_' + str(epoch) + '.png'
            m_utils.save_image(img_merge, filename)

        if (i + 1) % 10== 0:
            print('Validate: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RML={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'Rel={average.absrel:.3f}\n'
          'Log10={average.lg10:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    logger.add_scalar('Test/rmse', avg.rmse, epoch)
    logger.add_scalar('Test/Rel', avg.absrel, epoch)
    logger.add_scalar('Test/log10', avg.lg10, epoch)
    logger.add_scalar('Test/Delta1', avg.delta1, epoch)
    logger.add_scalar('Test/Delta2', avg.delta2, epoch)
    logger.add_scalar('Test/Delta3', avg.delta3, epoch)
    return avg, img_merge



if __name__ == '__main__':
    main()



