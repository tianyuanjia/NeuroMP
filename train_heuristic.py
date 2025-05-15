import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import pickle
from time import time
from algorithm.dijkstra import dijkstra
from diff_astar import HeuristicNeuralAstar
import torch.nn as nn
from environment import KukaEnv, MazeEnv, SnakeEnv, Kuka2Env
import argparse
import random
from str2name import str2name
import pandas as pd
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--is_training', action='store_true', default=True, help='Training or Testing.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=8, help='The batch size.')  # 8
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--epoch', type=int, default=400, help='The number of epochs.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--schedule', type=bool, default=True, help='Whether to turn on optimizer scheduler.')
parser.add_argument('--finetune', type=bool, default=False, help='Whether to finetune the model.')
parser.add_argument('--robot', type=str, default="kuka7", help='[maze3, ur5, kuka13, kuka14]')
parser.add_argument('--n_sample', type=int, default=100, help='The number of samples.')
parser.add_argument('--k', type=int, default=10, help='The number of neighbors.')
parser.add_argument('--gamma', type=int, default=0.0001, help='The weight of the cpa loss.')
args = parser.parse_args()
loop = 10
T = 0


def backtrack(start_n, goal_n, prev_node, num_nodes):
    path = torch.zeros(num_nodes).to(device)
    path[goal_n] = 1
    path[start_n] = 1
    loc = start_n
    while loc != goal_n:
        loc = prev_node[loc]
        path[loc] = 1
    return path


def compute_pathcost(path, states):
    path = path.astype(np.int_)
    cost = 0
    for i in range(0, path.shape[0] - 1):
        cost += np.linalg.norm(states[path[i]] - states[path[i + 1]])

    return cost


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def test_planning():
    indexes = np.arange(2000, 2500)
    pbar = tqdm(indexes)
    # pbar = indexes
    total_pathcost_astar = []
    model.eval()
    success = 0
    for index in pbar:
        pb = env.init_new_problem(index)
        if args.robot == "maze3":
            points, neighbors, edge_cost, edge_index, edge_free, _ = graphs[index]
        else:
            points, neighbors, edge_cost, edge_index, edge_free, _, _ = graphs[index]
        all_states = np.copy(points)
        goal_index = 1
        dist, prev = dijkstra(list(range(len(points))), neighbors, edge_cost, goal_index)
        start_index = 0

        edge_index = torch.LongTensor(edge_index.T)
        temp_tensor = torch.FloatTensor()
        node_free = temp_tensor.new_zeros(len(points), len(points))
        node_free[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_free).squeeze()

        edge_cost = sum(edge_cost.values(), [])
        edge_cost = np.array(edge_cost)
        cost_maps = temp_tensor.new_zeros(len(points), len(points))
        cost_maps[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_cost).squeeze()

        diag_node_free = torch.diag(node_free, 0)
        node_free = node_free.to(device)
        cost_maps = cost_maps.to(device)
        points = torch.FloatTensor(np.array(points)).to(device)
        edge_index = edge_index.to(device)

        labels = torch.zeros(len(points), 2)
        labels[:, 0] = 1  # nodes in the free space
        labels[goal_index, 0] = 0
        labels[goal_index, 1] = 1  # goal node
        labels = labels.to(device)

        current_loop = 5

        open_list, pred_history, pred_path, pred_ordered_path = model(start_index, goal_index, points, edge_index,
                                                                      node_free, cost_maps, current_loop, labels,training=False)

        if pred_ordered_path:  # if the results are not empty
            pred_ordered_path = torch.stack(pred_ordered_path).cpu().detach().numpy()
        pred_ordered_path = np.concatenate((pred_ordered_path, np.array([goal_index])), axis=0)

        if pred_path[start_index]:
            path_cost_astar = compute_pathcost(pred_ordered_path, all_states)
            success += 1
            total_pathcost_astar.append(path_cost_astar)
    return np.mean(total_pathcost_astar), success

def proto_align_loss(feat, feat_aug, temperature=0.3):
    cl_dim = feat.shape[0]

    feat_norm = torch.norm(feat, dim=-1, keepdim=True)
    feat = torch.div(feat, feat_norm)

    feat_aug_norm = torch.norm(feat_aug, dim=-1, keepdim=True)
    feat_aug = torch.div(feat_aug, feat_aug_norm)

    feat = feat.view(feat.size(0), -1)
    feat_aug = feat_aug.view(feat_aug.size(0), -1)

    sim_clean = torch.mm(feat, feat.t())
    mask = (torch.ones_like(sim_clean) - torch.eye(cl_dim, device=sim_clean.device)).bool()
    sim_clean = sim_clean.masked_select(mask).view(cl_dim, -1)

    sim_aug = torch.mm(feat, feat_aug.t())
    sim_aug = sim_aug.masked_select(mask).view(cl_dim, -1)

    logits_pos = torch.bmm(feat.view(cl_dim, 1, -1), feat_aug.view(cl_dim, -1, 1)).squeeze(-1)
    logits_neg = torch.cat([sim_clean, sim_aug], dim=1)

    logits = torch.cat([logits_pos, logits_neg], dim=1)
    instance_labels = torch.zeros(cl_dim).long().to(sim_clean.device)

    loss = F.cross_entropy(logits / temperature, instance_labels)

    return loss

if __name__ == '__main__':

    INFINITY = float('inf')
    writer = SummaryWriter()
    env, model, weights_name, _, _, data_path = str2name(args.robot, args.n_sample, args.k)
    model = model.to(device)
    if args.finetune:
        weights_name = "Astar_weights_w3/kuka13/train/{}_{}_{}_astar_13.pt".format(args.robot, args.n_sample, args.k)
        model.load_state_dict(torch.load(weights_name, map_location=device))

    with open(data_path, 'rb') as f:
        graphs = pickle.load(f)

    results = []
    if args.finetune:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    if args.schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion = nn.L1Loss()

    s_time = time()
    losses = 0
    loss_augs = 0
    loss_weaks = 0
    iteration = 0
    min_test_path_cost = 100
    current_loop = 5

    for iter_i in range(args.epoch):
        model.train()

        indexes = np.random.permutation(2000)

        pbar = tqdm(indexes)
        for index in pbar:
            pbar.set_description(f"Epochs {iter_i + 1}/{args.epoch}")
            pb = env.init_new_problem(index)

            if args.robot == "maze3":
                points, neighbors, edge_cost, edge_index, edge_free, _ = graphs[index]  # maze2
            else:
                points, neighbors, edge_cost, edge_index, edge_free, _, _ = graphs[index]


            goal_index = np.random.choice(len(points))
            dist, prev = dijkstra(list(range(len(points))), neighbors, edge_cost, goal_index)
            prev[goal_index] = goal_index
            valid_node = (np.array(list(dist.values())) != INFINITY)
            if sum(valid_node) == 1:
                continue
            start_index = np.random.choice(np.arange(len(valid_node))[valid_node])  # random select a start node

            edge_index = torch.LongTensor(edge_index.T)
            temp_tensor = torch.FloatTensor()
            node_free = temp_tensor.new_zeros(len(points), len(points))
            node_free[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_free).squeeze()

            edge_cost = sum(edge_cost.values(), [])
            edge_cost = np.array(edge_cost)

            cost_maps = temp_tensor.new_zeros(len(points), len(points))
            cost_maps[edge_index[0, :], edge_index[1, :]] = torch.FloatTensor(edge_cost).squeeze()

            labels = torch.zeros(len(points), 2)
            labels[:, 0] = 1  # nodes in the free space
            labels[goal_index, 0] = 0
            labels[goal_index, 1] = 1  # goal node

            node_free = node_free.to(device)
            cost_maps = cost_maps.to(device)
            labels = labels.to(device)
            points = np.stack(points)
            points = torch.FloatTensor(points).to(device)
            edge_index = edge_index.to(device)

            '''weak structure'''
            pred_weak, h_weak = model(start_index, goal_index, points, edge_index, node_free, cost_maps,
                                                    current_loop, labels)
            pred_history_weak, pred_path_weak = pred_weak

            '''aug structure'''
            node_aug = temp_tensor.new_zeros(len(points), len(points))
            node_aug[edge_index[0, :], edge_index[1, :]] = 1.0
            node_aug = torch.FloatTensor(node_aug).to(device)
            pred_aug, h_aug = model(start_index, goal_index, points, edge_index, node_aug, cost_maps,
                                                    current_loop, labels)
            pred_history_aug, pred_path_aug = pred_aug

            oracle_path = backtrack(start_index, goal_index, prev, points.size(0))  # dijkstra

            loss_aug = criterion(pred_history_aug, oracle_path) / points.size(0)
            loss_weak = criterion(pred_history_weak, oracle_path) / points.size(0)
            loss_cpa = proto_align_loss(h_weak, h_aug)
            loss = loss_aug + loss_weak + args.gamma * loss_cpa

            loss.backward()
            losses += loss.item()
            loss_augs += loss_aug.item()
            loss_weaks += loss_weak.item()

            if T % args.batch_size == 0:
                iteration += 1
                optimizer.step()
                optimizer.zero_grad()
                curr_loss = losses / iteration
                pbar.set_postfix({'curr_loss': curr_loss})
                writer.add_scalar('{}_{}_{}/total_loss'.format(args.robot, args.n_sample, args.k), curr_loss, iteration)
                writer.add_scalar('{}_{}_{}/aug_loss'.format(args.robot, args.n_sample, args.k), loss_augs/iteration, iteration)
                writer.add_scalar('{}_{}_{}/weak_loss'.format(args.robot, args.n_sample, args.k), loss_weaks/iteration,
                                  iteration)
            T += 1

        mean_test_path_cost = test_planning()
        if mean_test_path_cost < min_test_path_cost:
            min_test_path_cost = mean_test_path_cost
            print('The average path cost is:', min_test_path_cost)
            torch.save(model.state_dict(), weights_name)
        if args.schedule:
            scheduler.step()

    print('The total training time is', time() - s_time)
    writer.close()

