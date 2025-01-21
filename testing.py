import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from Emetrics import get_cindex, get_ci, get_rm2, get_mse, get_rmse, get_pearson, get_spearman
from DTADataset import *
from GNNNet import GNNNet
from Creating_Train_and_Test_set import create_test_dataset

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print(f'Make prediction for {len(loader.dataset)} samples...')
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def calculate_metrics(Y, P, dataset='davis'):
    cindex = get_cindex(Y, P)
    cindex2 = get_ci(Y, P)
    rm2 = get_rm2(Y, P)
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)

    print(f'Metrics for {dataset}:')
    print(f'cindex: {cindex}')
    print(f'cindex2: {cindex2}')
    print(f'rm2: {rm2}')
    print(f'mse: {mse}')
    print(f'pearson: {pearson}')

    result_file_name = os.path.join(results_path, f'result_{model_st}_{dataset}.txt')
    result_str = f'{dataset}\nrmse: {rmse} mse: {mse} pearson: {pearson} spearman: {spearman} ci: {cindex} rm2: {rm2}'
    print(result_str)
    with open(result_file_name, 'w') as file:
        file.write(result_str)

def plot_density(Y, P, dataset='davis'):
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.scatter(P, Y, color='blue', s=40)
    plt.title(f'density of {dataset}', fontsize=30, fontweight='bold')
    plt.xlabel('predicted', fontsize=30, fontweight='bold')
    plt.ylabel('measured', fontsize=30, fontweight='bold')
    plt.plot([5, 11], [5, 11], color='black')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(results_path, f'{dataset}.png'), dpi=500, bbox_inches='tight')

if __name__ == '__main__':
    dataset = 'davis'
    cuda_name = 'cuda:0'
    print(f'dataset: {dataset}')
    print(f'cuda_name: {cuda_name}')

    TEST_BATCH_SIZE = 512
    model_file_name = "path_to_the_model"  
    results_path = "path_to_the_result_dir"  

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

    result_file_name = os.path.join(results_path, f'result_{GNNNet.__name__}_{dataset}.txt')

    test_data = create_test_dataset(dataset)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = GNNNet()
    model.to(device)
    model.load_state_dict(torch.load(model_file_name, map_location=device))

    Y, P = predicting(model, device, test_loader)
    calculate_metrics(Y, P, dataset)
    plot_density(Y, P, dataset)
