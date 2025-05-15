import torch
from environment import MazeEnv, KukaEnv, Kuka2Env, UR5Env
import numpy as np
from astar import HeuristicNeuralAstar
from selector import CollisionNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def str2name(robot, n_sample, k):

    model_astar_path = "Astar_weights/{}_{}_{}.pt".format(robot, n_sample, k)
    model_coll_path = "collision_weights/{}_{}_{}_collision.pt".format(robot, n_sample, k)
    data_path = "data/own_pkl/{}_prm_{}_{}.pkl".format(robot, n_sample, k)

    if robot == 'maze3':
        env = MazeEnv(dim=3)
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=3, obs_size=2, embed_size=32)  # 64
        model_coll = CollisionNet(config_size=3, embed_size=64, obs_size=2, use_obstacles=True)

    elif robot == 'ur5':
        env = UR5Env()
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=3, obs_size=2, embed_size=32)  # 64
        model_coll = CollisionNet(config_size=3, embed_size=64, obs_size=2, use_obstacles=True)

    elif robot == 'kuka13':
        env = KukaEnv(kuka_file="kuka_iiwa/model_3.urdf", map_file="maze_files/kukas_13_3000.pkl")
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=13, obs_size=6, embed_size=32)
        model_coll = CollisionNet(config_size=13, embed_size=64, obs_size=6, use_obstacles=True)

    elif robot == 'kuka14':
        env = Kuka2Env()
        model_astar = HeuristicNeuralAstar(Tmax=1, config_size=14, obs_size=6, embed_size=32)  # 32
        model_coll = CollisionNet(config_size=14, embed_size=64, obs_size=6, use_obstacles=True)

    return env, model_astar, model_astar_path, model_coll, model_coll_path, data_path
