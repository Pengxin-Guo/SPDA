import torch
import torch.nn as nn
import math
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from argparse import ArgumentParser
from model.SPDA import SPDAModel
from utils.data_provider import load_images, load_images_list
from utils.utils import *
from tensorboardX import SummaryWriter


class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res
    

def evaluate(model_instance, target_dataloader, device, cur_step):
    model_instance.eval()
    num_iter = len(target_dataloader)
    iter_test = iter(target_dataloader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        inputs, labels = inputs.to(device), labels.to(device)       
        probabilities = model_instance.predict(inputs)
        probabilities = probabilities.data.float()
        labels = labels.data.float()
       
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels) / float(all_labels.size()[0])
    
    # writer.add_scalar('val/acc', accuracy, cur_step)
    
    return accuracy
    

def train(model_instance, device, train_source_loader, train_target_loader, group_ratios, optimizer, lr_scheduler, epoch):
    cur_step = epoch*len(train_source_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    # writer.add_scalar('train/lr', cur_lr, cur_step)
   
    model_instance.train()
    
    # using iter
    if args.loop_way == "iter":
        step_source, step_target = len(train_source_loader), len(train_target_loader)
        max_step = max(step_source, step_target)
        source_dataset, target_dataset = iter(train_source_loader), iter(train_target_loader)
        for step in range(max_step):
            try:
                source_inputs, source_labels = source_dataset.next()
            except:
                source_dataset =  iter(train_source_loader)
                source_inputs, source_labels = source_dataset.next()
            try:
                target_inputs, _ = target_dataset.next()
            except:
                target_dataset =  iter(train_target_loader)
                target_inputs, _ = target_dataset.next()
            
            source_inputs, source_labels = source_inputs.to(device, non_blocking=True), source_labels.to(device, non_blocking=True)
            target_inputs = target_inputs.to(device, non_blocking=True)

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, cur_step / 5)
            optimizer.zero_grad()

            source_outputs, target_outputs, domain_distance = model_instance(source_inputs, target_inputs, source_labels)

            class_criterion = nn.CrossEntropyLoss()
            classifier_loss = class_criterion(source_outputs, source_labels.long())

            lambd = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
            domain_loss = lambd * 2 * domain_distance
            final_loss = classifier_loss + domain_loss
            # writer.add_scalar('train/domain_distance', domain_distance.item(), cur_step)
            # writer.add_scalar('train/domain_loss', domain_loss.item(), cur_step)
            
            final_loss.backward()
            optimizer.step()
        
            cur_acc = accuracy(source_outputs, source_labels, topk=(1,))
            
            # writer.add_scalar('train/classifier_loss', classifier_loss.item(), cur_step)
            # writer.add_scalar('train/final_loss', final_loss.item(), cur_step)
            # writer.add_scalar('train/acc', cur_acc[0].item(), cur_step)
        
            cur_step += 1
    
    # using zip
    else:
        for (datas, datat) in zip(train_source_loader, train_target_loader):
            source_inputs, source_labels = datas
            target_inputs, _ = datat
        
            source_inputs, source_labels = source_inputs.to(device, non_blocking=True), source_labels.to(device, non_blocking=True)
            target_inputs = target_inputs.to(device, non_blocking=True)

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, cur_step / 5)
            optimizer.zero_grad()

            source_outputs, target_outputs, domain_distance = model_instance(source_inputs, target_inputs, source_labels)

            class_criterion = nn.CrossEntropyLoss()
            classifier_loss = class_criterion(source_outputs, source_labels.long())

            lambd = 2 / (1 + math.exp(-10 * (epoch) / args.epochs)) - 1
            domain_loss = lambd * 2 * domain_distance
            final_loss = classifier_loss + domain_loss
            # writer.add_scalar('train/domain_distance', domain_distance.item(), cur_step)
            # writer.add_scalar('train/domain_loss', domain_loss.item(), cur_step)

            final_loss.backward()
            optimizer.step()
        
            cur_acc = accuracy(source_outputs, source_labels, topk=(1,))
              
            # writer.add_scalar('train/classifier_loss', classifier_loss.item(), cur_step)
            # writer.add_scalar('train/final_loss', final_loss.item(), cur_step)
            # writer.add_scalar('train/acc', cur_acc[0].item(), cur_step)
        
            cur_step += 1


def select_target_data(model_instance, source_list, target_list):
# only add, with replacement
# means: update the pseudo labels at every epoch
    model_instance.eval()
    target_dataloader = load_images_list(target_list, batch_size=args.batch_size, num_workers=args.works, is_train=False)
    num_iter = len(target_dataloader)
    iter_test = iter(target_dataloader)
    add_num = 0
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        inputs = inputs.to(device)      
        probabilities = model_instance.predict(inputs)
        softmax = nn.Softmax(dim=1)
        softmax_prob = softmax(probabilities)
        # probabilities = probabilities.data.float()
        softmax_prob = softmax_prob.data.float()
        pred_pro, pred_label =torch.max(softmax_prob, dim=1)
        index = pred_pro>args.threshold
        # print(index)
        for j in range(len(index)):
            if index[j]:
                add_item = target_list[i*args.batch_size+j]
                add_item = add_item.split(' ')[0] + ' ' + str(pred_label[j].item())
                source_list.append(add_item)
                add_num += 1
    print("Add {} data !".format(add_num))
    return source_list
          
   
if __name__ == '__main__': 
    parser = ArgumentParser("SPDA")
    parser.add_argument('--name', help='log file name, auto generate')
    parser.add_argument("--batch-size", default=128, type=int, help='batch size')
    parser.add_argument("--epochs", default=200, type=int, help='# of training epochs')
    parser.add_argument("--print-frequency", default=1, type=int)
    parser.add_argument("--dst", default='office31', type=str, help='office31, officehome, visda')
    parser.add_argument("--source", default='amazon_31_list', type=str, help='source domain')
    parser.add_argument("--target", default='dslr_10_list', type=str, help='target domain')
    parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
    parser.add_argument("--works", default=4, type=int, help='# of works')
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--loop-way', type=str, default='zip', help='zip / iter')
    parser.add_argument('--threshold', type=float, default=0.9, help='select target data for self training')
    
    args = parser.parse_args()
    if args.name == None:
        # auto generate writer name
        args.name = args.source + "_" + args.target + "_lr_" +  str(args.lr) + "_" + args.loop_way + "_"+ str(args.epochs) + "_epoch"
    
    if args.dst == "office31":
        root_path = '../data/office31/'
        class_num = 31
    elif args.dst == "officehome":
        root_path = '../data/office-home/'
        class_num = 65
    elif args.dst == "visda":
        root_path = '../data/visda/'
        class_num = 12
    else:
        print('No support {} dataset'.format(args.dst))
        exit()
        
    print(args)
    
    # set gpu
    gpu = pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # tensorboard
    # writer = SummaryWriter(log_dir=os.path.join(os.path.join('../writer', args.name), "tb"))

    source_file = root_path + args.source + ".txt" 
    target_file = root_path + args.target + ".txt"
    
    model_instance = SPDAModel(base_net='ResNet50', hidden_dim=1024, class_num=class_num).to(device)
 
    train_source_loader = load_images(source_file, batch_size=args.batch_size, num_workers=args.works)
    train_target_loader = load_images(target_file, batch_size=args.batch_size, num_workers=args.works)
    test_target_loader = load_images(target_file, batch_size=args.batch_size, num_workers=args.works, is_train=False)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    optimizer = torch.optim.SGD(param_groups)
    lr_scheduler = INVScheduler(gamma=0.001, decay_rate=0.75, init_lr=args.lr)
    
    best_acc = 0
    print("start train...")
    Iter = 1
    
    source_list = open(source_file).readlines()
    target_list = open(target_file).readlines()

    for epoch in range(args.epochs):
        # training
        train(model_instance, device, train_source_loader, train_target_loader, group_ratios, optimizer, lr_scheduler, epoch)
        # validation
        cur_step = (epoch+1) * len(train_source_loader)
        eval_result = evaluate(model_instance, test_target_loader, device, cur_step)
        if epoch % args.print_frequency == 0:
            print("Valid: [{:3d}/{}] Acc {:.4%}".format(epoch+1, args.epochs, eval_result))
        if best_acc < eval_result:
            best_acc = eval_result
                
        # do self training
        temp_source_list = select_target_data(model_instance, source_list[:], target_list[:])
        train_source_loader = load_images_list(temp_source_list, batch_size=args.batch_size, num_workers=args.works)
        Iter += 1
    print('finish train\n')
    print("best test acc = ", best_acc)
