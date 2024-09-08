import numpy as np
import pandas as pd
from kgctf import StockTensorManager, KGTensorManager, FactorManager

# Load data
trn_file = '/home/duduuman/KG-CTF/data/stock_train.csv'
true_file = '/home/duduuman/KG-CTF/data/stock_true.csv'
missing_idxs = np.load('/home/duduuman/KG-CTF/data/missing_idxs.npy', allow_pickle=True).item()
aux_file = '/home/duduuman/KG-CTF/data/updated_relations.txt'
common_symbols_file = '/home/duduuman/KG-CTF/data/common_symbols2id.txt'

stock_train = pd.read_csv(trn_file, dtype={'ID': str})
stock_true = pd.read_csv(true_file, dtype={'ID': str})
common_symbols = pd.read_csv(common_symbols_file, sep='\t', header=None, names=['symbol', 'id'])
kg_data = pd.read_csv(aux_file, sep='\t', header=None, names=['head', 'relation', 'tail'])

rank = 10

# Initialize StockTensorManager and KGTensorManager
stock_manager = StockTensorManager(stock_train, common_symbols)
stock_manager_true = StockTensorManager(stock_true, common_symbols)
kg_manager = KGTensorManager(kg_data, common_symbols, rank)

# Create tensors
stock_tensors = stock_manager.create_stock_tensor_parallel()
stock_tensors_true = stock_manager_true.create_stock_tensor_parallel() 
kg_tensors, kg_entity_matrices, entity_embeddings = kg_manager.create_kg_tensor_parallel()

print(len(list(stock_tensors.keys())))

# Initialize factors
factors = FactorManager(stock_tensors, kg_tensors, entity_embeddings, common_symbols, kg_data, rank)
factors.initialize_factors()

print(factors)

# Function to calculate RMSE
def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Function to predict the stock tensors
def predict_stock_tensors(stock_tensors, factors_stock):
    predictions = {}

    stock_symbols = list(stock_tensors.keys())
    num_factors = len(factors_stock[0])

    for symbol, tensor in stock_tensors.items():
        print(symbol)
        print(tensor)

        idx = list(stock_tensors.keys()).index(symbol)

        print(idx)
        U_k = factors_stock[0][idx]
        S_k = factors_stock[1][idx]
        V = factors_stock[2]
        predicted_tensor = U_k @ S_k @ V.T
        predictions[symbol] = predicted_tensor
    return predictions

# Function to evaluate RMSE using the true and predicted tensors
def evaluate_rmse(stock_true, predicted_tensors, missing_idxs):
    rmse_values = []
    for symbol, indices in missing_idxs.items():
        true_tensor = stock_true[symbol]
        pred_tensor = predicted_tensors[symbol]
        for row, col in indices:
            if row < true_tensor.shape[0] and col < true_tensor.shape[1]:
                true_value = true_tensor[row, col]
                pred_value = pred_tensor[row, col]
                rmse_values.append((true_value, pred_value))
    true_values = np.array([x[0] for x in rmse_values])
    pred_values = np.array([x[1] for x in rmse_values])
    rmse = calculate_rmse(pred_values, true_values)
    return rmse

# Set parameters for ALS update
num_epochs = 10
lambdas = (10, 0.01, 1)

# Training loop
for epoch in range(num_epochs):
    beta_k = 0.3 if 5 <= epoch < 15 else 0.0
    factors_stock, factors_kg = factors.als_update(lambdas, beta_k)
    
    # After updating, make predictions and calculate RMSE
    predicted_tensors = predict_stock_tensors(stock_tensors, factors_stock)
    rmse = evaluate_rmse(stock_tensors_true, predicted_tensors, missing_idxs)
    
    print(f"Epoch {epoch+1}/{num_epochs} complete. RMSE: {rmse:.4f}")

print("Training complete!")
