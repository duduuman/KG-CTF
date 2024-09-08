import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.linalg import solve_sylvester
from collections import OrderedDict

# StockTensorManager to handle stock tensors
class StockTensorManager:
    def __init__(self, stock_data, common_symbols):
        self.stock_data = stock_data[stock_data['ID'].isin(common_symbols['symbol'])]
        self.common_symbols = common_symbols
        self.stock_tensors = None

    def create_single_stock_tensor(self, args):
        symbol, group, features = args
        unique_dates = sorted(group['Date'].unique())
        tensor = np.zeros((len(unique_dates), len(features)))
        date_idx = {date: idx for idx, date in enumerate(unique_dates)}
        group = group.set_index('Date')
        for date, row in group.iterrows():
            tensor[date_idx[date], :] = row[features].values
        return symbol, tensor

    def create_stock_tensor_parallel(self):
        features = [col for col in self.stock_data.columns if col not in ['ID','Date', 'volatility_kchi', 'volatility_kcli', 'trend_psar_down', 'trend_psar_down_indicator']]
        all_symbols = sorted(self.stock_data['ID'].unique())
        common_stock_indices = [all_symbols.index(symbol) for symbol in self.common_symbols if symbol in all_symbols]
        grouped = self.stock_data.groupby('ID')
        result = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(self.create_single_stock_tensor)((symbol, group, features)) for symbol, group in tqdm(grouped)
        )
        self.stock_tensors = dict(result)
        return self.stock_tensors


# KGTensorManager to handle knowledge graph tensors
class KGTensorManager:
    def __init__(self, kg_data, common_symbols, rank):
        self.kg_data = kg_data[kg_data['tail'].isin(common_symbols['id'])]
        self.common_symbols = common_symbols
        self.kg_tensors = None
        self.rank = rank
        self.entity_embeddings = {entity: np.random.rand(self.rank) for entity in kg_data['head'].unique()}


    def create_single_kg_tensor(self, args):
        symbol_id, group, fixed_relations = args
        unique_entities = sorted(set(group['head'].unique()))
        entity_idx = {entity: idx for idx, entity in enumerate(unique_entities)}
        relation_idx = {relation: idx for idx, relation in enumerate(fixed_relations)}
        tensor = np.zeros((len(unique_entities), len(fixed_relations)))
        for _, row in group.iterrows():
            head = row['head']
            relation = row['relation']
            tensor[entity_idx[head], relation_idx[relation]] = 1

        M_k = np.array([self.entity_embeddings[entity] for entity in unique_entities])  # Example of entity embedding matrix
        
        return symbol_id, tensor, M_k

    def create_kg_tensor_parallel(self):
        all_ids = self.common_symbols['id'].unique()
        grouped = self.kg_data.groupby('tail')
        fixed_relations = sorted(self.kg_data['relation'].unique())
        result = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(self.create_single_kg_tensor)((symbol_id, group, fixed_relations))
            for symbol_id, group in tqdm(grouped) if symbol_id in self.common_symbols['id'].unique()
        )
        self.kg_tensors = {res[0]: res[1] for res in result}
        self.kg_entity_matrices = {res[0]: res[2] for res in result}
        return self.kg_tensors, self.kg_entity_matrices, self.entity_embeddings


# FactorManager to manage ALS updates and factor initialization
class FactorManager:
    def __init__(self, stock_tensors, kg_tensors, entity_embeddings, common_symbols, kg_data, rank):
        self.common_symbols = common_symbols
        self.kg_data = kg_data 

        self.symbol_to_id = dict(zip(self.common_symbols['symbol'], self.common_symbols['id']))
        self.id_to_symbol = {v: k for k, v in self.symbol_to_id.items()}

        ordered_stock_tensors = OrderedDict()
        ordered_kg_tensors = OrderedDict()

        for symbol in stock_tensors.keys():
            symbol_id = self.symbol_to_id[symbol]  
            #if symbol_id in kg_tensors:  
            ordered_stock_tensors[symbol] = stock_tensors[symbol]
            ordered_kg_tensors[symbol_id] = kg_tensors[symbol_id]

        self.stock_tensors = ordered_stock_tensors
        self.kg_tensors = ordered_kg_tensors
        self.entity_embeddings = entity_embeddings

        self.rank = rank
        self.factors_stock = None
        self.factors_kg = None

    def initialize_factors(self):
        def xavier_init(shape):
            return np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1]))

        num_companies = len(self.stock_tensors)
        stock_tensor_shapes = [tensor.shape for tensor in self.stock_tensors.values()]
        kg_tensor_shapes = [tensor.shape for tensor in self.kg_tensors.values()]

        # Initialize factors for stock data
        self.factors_stock = [
            [xavier_init((shape[0], self.rank)) for shape in stock_tensor_shapes],  # U_k
            [np.diag(np.random.rand(self.rank)) for _ in range(num_companies)],  # S_k
            xavier_init((stock_tensor_shapes[0][1], self.rank)),  # V
            [xavier_init((shape[0], self.rank)) for shape in stock_tensor_shapes],  # Q_k
            xavier_init((self.rank, self.rank))  # H
        ]

        # Initialize factors for knowledge graph data
        self.factors_kg = [
            [xavier_init((shape[0], self.rank)) for shape in kg_tensor_shapes],  # M_k
            [xavier_init((shape[0], self.rank)) for shape in kg_tensor_shapes],  # placeholder for future use
            xavier_init((kg_tensor_shapes[0][1], self.rank)),  # R
            xavier_init((self.rank, self.rank)),  # placeholder for future use
            [np.diag(np.sum(G_k, axis=0)) for G_k in self.kg_tensors.values()],  # D_k
            np.ones((kg_tensor_shapes[0][1], self.rank))  # placeholder for future use
        ]
        return self.factors_stock, self.factors_kg

    # Function to update U_k
    def update_U_k(self, T_k, V, S_k, Q_k, H, lambda_u, U_k_old, beta_k):
        U_k = (T_k @ V @ S_k + lambda_u * Q_k @ H) @ np.linalg.inv(S_k @ V.T @ V @ S_k + lambda_u * np.eye(self.rank))
        U_k += beta_k * (U_k - U_k_old)  # Apply momentum
        return U_k

    # Function to update S_k
    def update_S_k(self, T_k, U_k, V, G_k, M_k, R, D_k, lambda_l, lambda_r, S_k_old, beta_k):
        D_k_inv = np.diag(np.where(np.diag(D_k) != 0, 1 / np.diag(D_k), 0))
        S_k_num = (U_k.T @ T_k @ V) + (M_k.T @ G_k @ R) + lambda_r * (np.ones((self.rank, G_k.shape[1])) @ R) + lambda_r * (np.ones((self.rank, G_k.shape[1])) @ D_k_inv @ G_k.T @ M_k)
        S_k_den = ((U_k.T @ U_k) * (V.T @ V)) + ((M_k.T @ M_k) * (R.T @ R)) + lambda_l * np.eye(self.rank) + lambda_r * (np.ones((self.rank, G_k.shape[1])) @ np.ones((G_k.shape[1], self.rank)))
        S_k = np.diag(np.diag(S_k_num @ np.linalg.pinv(S_k_den)))
        S_k += beta_k * (S_k - S_k_old)  # Apply momentum
        return S_k

    # Function to update V
    def update_V(self, lambda_l, V_old, beta_k):
        V_numerator = np.zeros((self.factors_stock[2].shape[0], self.factors_stock[2].shape[1]))
        V_denominator = np.zeros((self.factors_stock[2].shape[1], self.factors_stock[2].shape[1]))
        for idx, T_k in enumerate(self.stock_tensors.values()):
            U_k = self.factors_stock[0][idx]
            S_k = self.factors_stock[1][idx]
            V_numerator += T_k.T @ U_k @ S_k
            V_denominator += S_k @ U_k.T @ U_k @ S_k
        V = V_numerator @ np.linalg.inv(V_denominator + lambda_l * np.eye(self.rank))
        V += beta_k * (V - self.factors_stock[2])  # Apply momentum
        return V

    # Function to update M_k
    def update_M_k(self, G_k, M_k, S_k, R, D_k, lambda_r, lambda_l, M_k_old, beta_k):
        D_k_inv = np.diag(np.where(np.diag(D_k) != 0, 1 / np.diag(D_k), 0))
        A = lambda_r * ((G_k @ D_k_inv.T @ D_k_inv @ G_k.T)) + lambda_l * np.eye(M_k.shape[0])
        B = S_k @ R.T @ R @ S_k
        C = G_k @ R @ S_k + (lambda_r * G_k @ D_k_inv @ np.ones((G_k.shape[1], self.rank)) @ S_k) - (lambda_r * G_k @ D_k_inv @ R)
        M_k = solve_sylvester(A, B, C)
        M_k += beta_k * (M_k - M_k_old)  # Apply momentum
        return M_k

    # Function to update R
    def update_R(self, lambda_l, lambda_r, beta_k):
        R_numerator = np.zeros((self.factors_kg[2].shape[0], self.factors_kg[2].shape[1]))
        R_denominator = np.zeros((self.factors_kg[2].shape[1], self.factors_kg[2].shape[1]))
        for idx, G_k in enumerate(self.kg_tensors.values()):
            M_k = self.factors_kg[0][idx]
            S_k = self.factors_stock[1][idx]
            D_k = self.factors_kg[4][idx]
            D_k_inv = np.diag(np.where(np.diag(D_k) != 0, 1 / np.diag(D_k), 0))
            R_numerator += G_k.T @ M_k @ S_k + lambda_r * ((np.ones((G_k.shape[1], self.rank))) @ S_k - D_k_inv @ G_k.T @ M_k)  
            R_denominator += S_k @ M_k.T @ M_k @ S_k
        R = R_numerator @ np.linalg.inv(R_denominator + (lambda_l + lambda_r * len(self.stock_tensors)) * np.eye(self.rank))
        R += beta_k * (R - self.factors_kg[2])  # Apply momentum
        return R

    def update_QH(self, U_dict, H):
        K = len(U_dict)
        R = H.shape[0]
        Q_update = []
        H_tmp = np.zeros((K, R, R))
        H_update = np.zeros((R, R))

        for idx, U_k in U_dict.items():
            Z, _, P = np.linalg.svd(U_k @ H.T, full_matrices=False)
            Q_update.append((idx, Z @ P))
            H_update += Q_update[-1][1].T @ U_k

        H_update /= K
        return Q_update, H_update

    # ALS update method for each epoch
    def als_update(self, lambdas, beta_k):
        lambda_u, lambda_r, lambda_l = lambdas

        U_dict = {}

        # Step 1: Update U_k, S_k for each stock tensor
        for idx, symbol in enumerate(self.stock_tensors.keys()):
            try:
                T_k = self.stock_tensors[symbol]
                V = self.factors_stock[2]
                Q_k = self.factors_stock[3][idx]
                H = self.factors_stock[4]

                symbol_id = self.symbol_to_id[symbol]  # Assuming symbol ID is the key in tensors
                G_k = self.kg_tensors.get(symbol_id)

                # Update U_k
                S_k = self.factors_stock[1][idx]
                U_k_old = self.factors_stock[0][idx]
                U_k = self.update_U_k(T_k, V, S_k, Q_k, H, lambda_u, U_k_old, beta_k)
                self.factors_stock[0][idx] = U_k
                U_dict[idx] = U_k

                # Update S_k
                S_k_old = S_k
                if symbol_id in self.kg_tensors:
                    M_k = self.factors_kg[0][idx]
                    R = self.factors_kg[2]
                    D_k = self.factors_kg[4][idx]
                    # print(f"U_k.T shape: {U_k.T.shape}")
                    # print(f"T_k shape: {T_k.shape}")
                    # print(f"V shape: {V.shape}")
                    # print(f"M_k.T shape: {M_k.T.shape}")
                    # print(f"G_k shape: {G_k.shape}")
                    # print(f"R shape: {R.shape}")
                    S_k = self.update_S_k(T_k, U_k, V, G_k, M_k, R, D_k, lambda_l, lambda_r, S_k_old, beta_k)
                    self.factors_stock[1][idx] = S_k

                # Update M_k
                if symbol_id in self.kg_tensors:
                    M_k_old = M_k
                    M_k = self.update_M_k(G_k, M_k, S_k, R, D_k, lambda_r, lambda_l, M_k_old, beta_k)
                    self.factors_kg[0][idx] = M_k
                    
                    unique_entities = sorted(set(self.kg_data[self.kg_data['tail'] == symbol_id]['head'].unique()))
                    entity_idx = {entity: idx for idx, entity in enumerate(unique_entities)}
                    for entity, i in entity_idx.items():
                        self.entity_embeddings[entity] = M_k[i]
            except Exception as e:
                print(f"Error processing symbol {symbol}. Skipping. Error: {e}")
                continue
        # Step 2: Update V
        V_old = self.factors_stock[2]
        self.factors_stock[2] = self.update_V(self.stock_tensors, self.factors_stock, lambda_l, V_old, beta_k)

        # Step 3: Update R
        R_old = self.factors_kg[2]
        self.factors_kg[2] = self.update_R(self.kg_tensors, self.factors_kg, self.factors_stock, lambda_l, lambda_r, R_old, beta_k)

        # Step 4: Update Q_k and H using U_dict
        Q_update, H_update = self.update_QH(U_dict, self.factors_stock[4])
        for idx, Q_k in Q_update:
            self.factors_stock[3][idx] = Q_k
        self.factors_stock[4] = H_update

        # Step 5: Return updated factors
        return self.factors_stock, self.factors_kg


