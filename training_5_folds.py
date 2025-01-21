from gnn import GNNNet
from paste import *
from emetrics import *
from DTADataset import create_dataset_for_5folds

dataset = 'davis'
folds = [0, 1, 2, 3, 4]  

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
NUM_EPOCHS = 2000

print('Dataset: ', dataset)
print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = '/content/drive/MyDrive/models_KIBA'
results_dir = '/content/drive/MyDrive/results_KIBA'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Force using CPU
print('Using device:', device)

for fold in folds:

    model = GNNNet()
    model.to(device)
    model_st = GNNNet.__name__
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    start_epoch = 0

    print(f"Training for fold {fold}...")

    train_data, valid_data = create_dataset_for_5folds(dataset, fold)
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    best_mse = float('inf')
    best_epoch = -1
    model_file_name = os.path.join(models_dir, f'model_{model_st}_{dataset}_{fold}.model')

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        print('Predicting for validation data...')
        G, P = predicting(model, device, valid_loader)
        val_mse = get_mse(G, P)
        print(f'Epoch {epoch + 1} - Validation MSE: {val_mse}, Best MSE: {best_mse}')
        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print(f'MSE improved at epoch {best_epoch}; Best MSE: {best_mse}')
        else:
            print(f'No improvement since epoch {best_epoch}; Best MSE: {best_mse}')

