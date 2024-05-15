# from sklearn import linear_model, model_selection
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import set_seed
from torch.nn import functional as F


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):
    # data_loader = torch.utils.data.DataLoader(
    #     data_loader.dataset, batch_size=1, shuffle=False
    # )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            # batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)

def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r

def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    # clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()

# def compute_losses(net, loader):
#     criterion = nn.CrossEntropyLoss(reduction="none")
#     all_losses = []

#     for inputs, y in loader:
#         targets = y
#         inputs, targets = inputs.cuda(), targets.cuda()

#         logits, *_ = net(inputs)

#         losses = criterion(logits, targets).cpu().detach().numpy()
#         for l in losses:
#             all_losses.append(l)

#     return np.array(all_losses)

# def simple_mia(sample_loss, members, n_splits=10, random_state=0):
#     unique_members = np.unique(members)
#     if not np.all(unique_members == np.array([0, 1])):
#         raise ValueError("members should only have 0 and 1s")

#     attack_model = linear_model.LogisticRegression()
#     cv = model_selection.StratifiedShuffleSplit(
#         n_splits=n_splits, random_state=random_state
#     )
#     return model_selection.cross_val_score(
#         attack_model, sample_loss, members, cv=cv, scoring="accuracy"
#     )

# def cal_mia(model, forget_dataloader_test,  unseen_dataloader):
#     model.eval()

#     forget_losses = compute_losses(model, forget_dataloader_test)
#     unseen_losses = compute_losses(model, unseen_dataloader)

#     np.random.shuffle(forget_losses)
#     unseen_losses = unseen_losses[: len(forget_losses )]
    
#     samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
#     labels_mia = [0] * len(unseen_losses) + [1] * len(forget_losses)

#     mia_scores = simple_mia(samples_mia, labels_mia)
#     forgetting_score = abs(0.5 - mia_scores.mean())
    
#     model.train()

#     return {'MIA': mia_scores.mean(), 'Forgeting Score': forgetting_score}