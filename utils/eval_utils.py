from tqdm import tqdm
import numpy as np
import wandb
import torch


##########################
### Evaluate on groups ###
##########################

def evaluate_groups(args, model, loader, epoch=None, split='val', n_samples_per_dist=None):
    """ Test model on groups and log to wandb

        Separate script for femnist for speed."""

    groups = []
    num_examples = []
    accuracies = np.zeros(max(loader.dataset.groups)+1)

    model.eval()

    if n_samples_per_dist is None:
        n_samples_per_dist = args.n_test_per_dist

    for i, group in tqdm(enumerate(loader.dataset.groups), desc='Evaluating', total=len(loader.dataset.groups)):
        dist_id = group

        preds_all = []
        labels_all = []

        example_ids = np.nonzero(loader.dataset.group_ids == group)[0]
        example_ids = example_ids[np.random.permutation(len(example_ids))] # Shuffle example ids

        # Create batches
        batches = []
        X, Y, G = [], [], []
        counter = 0
        
        maxnum = int((len(example_ids) // (args.support_size * args.context_num)) * args.support_size * args.context_num)
       
        for i, idx in enumerate(example_ids):
            if i == (n_samples_per_dist - 1) or i == maxnum:
                break
                
            x, y, g = loader.dataset[idx]
            X.append(x); Y.append(y); G.append(g)
            if (i + 1) % (args.support_size * args.context_num) == 0:
                X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
                batches.append((X, Y, G))
                X, Y, G = [], [], []
        if X:
            X, Y, G = torch.stack(X), torch.tensor(Y, dtype=torch.long), torch.tensor(G, dtype=torch.long)
            batches.append((X, Y, G))

        counter = 0
        for images, labels, group_id in batches:
            
            images = torch.cat([images, images], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            
            labels = labels.detach().numpy()
            images = images.to(args.device)

            logits = model(images)
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)

            preds_all.append(preds)
            labels_all.append(labels)
            counter += len(images)

            if counter >= n_samples_per_dist:
                break

        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)

        # Evaluate
        accuracy = np.mean(preds_all == labels_all)

        num_examples.append(len(preds_all))
        accuracies[dist_id] = accuracy
        groups.append(dist_id)

        if args.log_wandb:
            if epoch is None:
                wandb.log({f"{split}_acc": accuracy, # Gives us Acc vs Group Id
                           "dist_id": dist_id})
            else:
                wandb.log({f"{split}_acc_e{epoch}": accuracy, # Gives us Acc vs Group Id
                           "dist_id": dist_id})

    # Log worst and average
    worst_case_acc = np.amin(accuracies)
    worst_case_group_size = num_examples[np.argmin(accuracies)]

    num_examples = np.array(num_examples)
    props = num_examples / num_examples.sum()
    average_case_acc = np.sum(accuracies)/len(loader.dataset.groups)

    total_size = num_examples.sum()

    stats = {f'worst_case_{split}_acc': worst_case_acc,
            f'worst_case_group_size_{split}': worst_case_group_size,
            f'average_{split}_acc': average_case_acc,
            f'total_size_{split}': total_size}

    if epoch is not None:
        stats['epoch'] = epoch

    if args.log_wandb:
        wandb.log(stats)

    return average_case_acc, worst_case_acc, stats


##########################################
### Evalauate on mixture distributions ###
##########################################


def eval_dist(dist, model, loader, args, n_samples_per_dist):
    """ Evaluates model on a specific sub distribution """
    preds_all = []
    labels_all = []

    # Set the correct test distribution
    loader.sampler.set_sub_dist(dist)

    counter = 0
    for images, labels, group_id in loader:
        
        images = torch.cat([images, images], dim=0)
        labels = torch.cat([labels, labels], dim=0)
            
        labels = labels.detach().numpy()
        images = images.to(args.device)
            

        logits = model(images)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)

        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)

        preds_all.append(preds)
        labels_all.append(labels)
        counter += len(images)

        logits = model(images).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)

        preds_all.append(preds)
        labels_all.append(labels)
        counter += len(images)

        if counter >= n_samples_per_dist:
            break

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    return preds_all, labels_all


def eval_dists_fn(split, args, model, loader, dists, n_samples_per_dist=1500):
    """ Evaluates modl on a set of mixture distributions """
    accuracies = np.zeros(len(dists))
    counter = 0
    for dist_id, dist in enumerate(tqdm(dists, f"{split}_evaluation")):

        preds_all, labels_all = eval_dist(dist, model, loader, args, n_samples_per_dist)
        accuracies[dist_id] = np.mean(preds_all == labels_all)
        counter += len(labels_all)

    return accuracies


def evaluate_each_corner(args, model, loader, split):
    """
        Assumes that loader returns deterministically all examples in the split in order.
        Note: There is no binning here.

    """

    preds_all = []
    labels_all = []
    group_ids_all = []


    counter = 0
    for images, labels, group_id in loader:

        labels = labels.detach().numpy()
        images = images.to(args.device)

        logits = model(images).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)

        preds_all.append(preds)
        labels_all.append(labels)
        group_ids_all.append(group_id.numpy())
        counter += len(images)

    # Empirical
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    group_ids_all = np.concatenate(group_ids_all)

    is_correct = preds_all == labels_all

    empirical_acc = np.mean(is_correct)

    group1_acc = np.mean(is_correct[group_ids_all == 0])
    group2_acc = np.mean(is_correct[group_ids_all == 1])
    group3_acc = np.mean(is_correct[group_ids_all == 2])
    group4_acc = np.mean(is_correct[group_ids_all == 3])

    worst_case_acc = np.amin([group1_acc, group2_acc, group3_acc, group4_acc])
    avg_case_acc = np.mean([group1_acc, group2_acc, group3_acc, group4_acc])

    stats = {f"worst_case_{split}_all_acc": worst_case_acc,
               f"avg_acc_{split}": avg_case_acc,
               f"empirical_{split}_acc": empirical_acc}

    ### Log wandb ###
    if args.log_wandb:
        for dist_id, accuracy in enumerate([group1_acc, group2_acc, group3_acc, group4_acc]):
            wandb.log({f"{split}_acc": accuracy, # Gives us Acc vs Group Id
                       "dist_id": dist_id})

        wandb.log(stats)

    return stats

def evaluate_mixtures(args, model, loader, dists, epoch=None, split='val'):
    """ Test model on a set of sub distributions (dists) and log to WandB"""

    groups = []
    dfs = []
    num_examples = []

    model.eval()

    n_samples_per_dist = args.n_test_per_dist

    accuracies = eval_dists_fn(split, args, model, loader, dists, n_samples_per_dist=n_samples_per_dist)


    ### Log wandb ###
    if args.log_wandb:
        for dist_id, accuracy in enumerate(accuracies):
            if epoch is not None:
                wandb.log({f"{split}_acc_e_{epoch}": accuracy, # Gives us Acc vs Group Id
                           "dist_id": dist_id})
            else:
                wandb.log({f"{split}_acc": accuracy, # Gives us Acc vs Group Id
                           "dist_id": dist_id})

    # Log worst, average and empirical accuracy
    worst_case_acc = np.amin(accuracies)

    avg_acc = np.mean(accuracies[:-1]) # Skip the empirical.

    empirical_case_acc = accuracies[-1]
    stats = {f"worst_case_{split}_all_acc": worst_case_acc,
               f"avg_{split}_acc": avg_acc,
               f"empirical_{split}_acc": empirical_case_acc}

    if epoch is not None:
        stats['epoch'] = epoch

    if args.log_wandb:
        wandb.log(stats)

    return worst_case_acc, avg_acc, empirical_case_acc, stats
