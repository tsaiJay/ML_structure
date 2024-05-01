from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from dataset import datasetSelector
from transform import TransformSelector
from model import LSTMModel


def train_process(args, epoch, model, train_loader, optimizer, criterion, device):

    model.to(device)
    model.train()

    total = 0
    correct = 0

    train_iterator = tqdm(train_loader, desc='train start...')
    for iter_idx, (datas, labels) in enumerate(train_iterator):

        optimizer.zero_grad()
        datas, labels = datas.to(device), labels.to(device)

        outputs = model(datas)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum()
        total += datas.size(0)

        train_acc = correct / total * 100

        train_iterator.set_description(
            f'train-> ID[{args.run_id}] CE Loss:{batch_loss:.5f} ACC: {train_acc:.3f}%')

        if args.use_wandb and iter_idx % args.log_freq == 0:
            wandb.log({
                'train_loss': batch_loss,
                'train_acc': train_acc, 
                'epoch': epoch})

    if args.lr_scheduler != 'none':
        scheduler.step()

    return train_acc


@torch.no_grad()
def test_process(model, test_loader, criterion, device):

    total = 0
    correct = 0

    model.to(device)
    model.eval()

    test_iterator = tqdm(test_loader, desc='testing')
    for datas, labels in test_iterator:
        datas, labels = datas.to(device), labels.to(device)
        
        outputs = model(datas)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum()
        total += labels.size(0)

        # test_iterator.set_description(
        #     f'test -> ID[{args.run_id}] Loss:{test_loss:.5f}')

        test_acc = correct / total * 100

    return test_acc


def main(args):
    save_dir = './records/' + args.run_name + '/'
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            name=args.run_name,
            config=args
        )

    print('-' * 20)
    train_transform, test_transform = TransformSelector(args.input_size)
    train_dataset, test_dataset = datasetSelector(args, train_transform, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    print('number of data(train, test): ({}, {})'.format(len(train_dataset), len(test_dataset)))

    print('-' * 20)
    DEVICE = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    print('model build [{}]...'.format(args.model))
    model = LSTMModel(
        input_size=args.img_size, 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        output_size=args.output_size
    )

    criterion = nn.CrossEntropyLoss()

    resume_ep = 0
    if args.resume:
        weight_path = './records/1/lstm_ep500_last.pth'
        resume_ep = torch.load(weight_path)['epoch']
        weight = torch.load(weight_path)['state_dict']
        model.load_state_dict(weight)
        test_result = test_process(model, test_loader, criterion, DEVICE)
        print(f'From epoch {resume_ep}, last state {test_result}')

    print('-' * 20)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    best_test_acc = 0.0
    best_epoch_num = 0
    epochs = args.num_epochs
    for epoch in range(resume_ep + 1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        train_acc = train_process(args, epoch, model, train_loader, optimizer, criterion, DEVICE)
        test_acc = test_process(model, test_loader, criterion, DEVICE)

        ckp = {
            'model_name': args.model,
            'state_dict': model.state_dict(),
            'epoch': epoch
        }

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch_num = epoch
            torch.save(ckp, save_dir + '{}_best.pth'.format(args.model))

        if (epoch) % args.ckp_epochs == 0:
            torch.save(ckp, save_dir + '{}_ep{}.pth'.format(args.model, epoch))

        torch.save(ckp, save_dir + '{}_ep{}_last.pth'.format(args.model, epochs))

        if args.use_wandb:
            wandb.log({
                # 'train_acc': train_acc,
                'test_acc': test_acc,
                'best_test_acc': best_test_acc,
                'best_epoch_num': best_epoch_num,
                'epoch': epoch})

        print('----> Train_acc:{:.4f}, Test_acc:{:.4f}, Best Test_acc:{:.4f}(ep{})'
            .format(train_acc, test_acc, best_test_acc, best_epoch_num))


if __name__ == '__main__':
    from config_loader import get_arg, build_record_folder

    args = get_arg()

    args.run_name = args.run_id

    if args.record:
        build_record_folder(args)
    else:
        print('=' * 26)
        print('= debug mode!! no record =')
        print('=' * 26)

    print('project name:', args.project_name)
    print('run name:', args.run_name)

    main(args)
