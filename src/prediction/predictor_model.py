import os
import sys
import warnings
import math

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
from torch.nn import GRU, LSTM, RNN, ReLU, Linear, Module, MSELoss, Tanh
from torch.utils.data import Dataset, DataLoader

# Check for GPU availability
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("device used: ", device)

PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
HISTORY_FNAME = "history.json"
COST_THRESHOLD = float("inf")


def get_patience_factor(N):
    # magic number - just picked through trial and error
    if N < 100:
        return 30
    patience = max(3, int(50 - math.log(N, 1.25)))
    return patience


def get_loss(model, device, data_loader, loss_function):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for data in data_loader:
            X, y = data[0].to(device), data[1].to(device)
            output = model(X)
            loss = loss_function(y, output)
            loss_total += loss.item()
    return loss_total / len(data_loader)


class CustomDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)



class Net(Module):
    def __init__(self, feat_dim, latent_dim, n_rnnlayers, decode_len, activation, rnn_unit, bidirectional):
        super(Net, self).__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.n_rnnlayers = n_rnnlayers
        self.decode_len = decode_len
        self.activation = activation
        self.rnn_unit = rnn_unit.lower()
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bool(self.bidirectional) else 1 
        
        self.rnn = self._get_rnn_unit()(
            input_size=self.feat_dim,
            hidden_size=self.latent_dim,
            num_layers=self.n_rnnlayers,
            bidirectional=bool(self.bidirectional),
            batch_first=True
            )
        self.relu_layer = self.get_activation()
        self.fc = Linear(self.num_directions*self.latent_dim, 1)

    def forward(self, X):
        # initial hidden states
        initial_state = self._get_hidden_initial_state(X)
        x, _ = self.rnn(X, initial_state)
        x = x[:, -self.decode_len:, :]
        x = self.relu_layer(x)  
        x = self.fc(x)
        x = torch.squeeze(x, dim=-1)        
        return x

    def _get_hidden_initial_state(self, X):
        if self.rnn_unit == 'lstm': 
            h0 = torch.zeros(
                self.n_rnnlayers*self.num_directions, X.size(0), self.latent_dim).to(device)
            c0 = torch.zeros(
                self.n_rnnlayers*self.num_directions, X.size(0), self.latent_dim).to(device)
            return ( h0, c0 )
        elif self.rnn_unit == 'gru' or self.rnn_unit == 'simple':
            h0 = torch.zeros(
                self.n_rnnlayers*self.num_directions, X.size(0), self.latent_dim).to(device)
            return h0
        else:
            raise Exception(
                f"Unrecognized rnn unit {self.rnn_unit}. Must be one of [ 'lstm', 'gru', 'simple']")

    def _get_rnn_unit(self): 
        if self.rnn_unit == 'lstm': 
            return LSTM
        elif self.rnn_unit == 'gru':
            return GRU
        elif self.rnn_unit == 'simple':
            return RNN
        else:
            raise Exception(
                f"Unrecognized rnn unit {self.rnn_unit}. Must be one of [ 'lstm', 'gru', 'simple']")

    def get_num_parameters(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):             
                nn = nn*s
            pp += nn
        return pp

    def get_activation(self):
        if self.activation == 'relu':
            return ReLU()
        elif self.activation == 'tanh':
            return Tanh()
        else:
            raise ValueError(
                f"Activation {self.activation} is unrecognized. Must be either 'tanh' or 'relu'.")



class Forecaster:
    """CNN Timeseries Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """
    MODEL_NAME = "CNN_Timeseries_Forecaster"

    def __init__(
            self,
            encode_len:int,
            decode_len:int,
            feat_dim:int,
            latent_dim:int,
            activation:str,
            rnn_unit:str,
            n_rnnlayers:int,
            bidirectional:bool,
            **kwargs
        ):
        """Construct a new CNN Forecaster."""
        self.encode_len = encode_len
        self.decode_len = decode_len
        self.feat_dim = feat_dim
        self.rnn_unit = rnn_unit
        self.n_rnnlayers = n_rnnlayers
        self.latent_dim = latent_dim
        self.activation = activation
        self.bidirectional = bidirectional
        self.batch_size = None  # calculated based on size of train data

        self.net = Net(
            feat_dim = self.feat_dim,
            latent_dim = self.latent_dim,
            n_rnnlayers = self.n_rnnlayers,
            decode_len = self.decode_len,
            activation = self.activation,
            rnn_unit = self.rnn_unit,
            bidirectional = self.bidirectional,
        )
        self.net.to(device)
        # print(self.net.get_num_parameters()) ; sys.exit()
        self.criterion = MSELoss()
        self.optimizer = optim.Adam( self.net.parameters() )
        self.print_period = 1

    def _get_X_and_y(self, data: np.ndarray, is_train:bool=True) -> np.ndarray:
        """Extract X (historical target series), y (forecast window target) 
            When is_train is True, data contains both history and forecast windows.
            When False, only history is contained.
        """
        N, T, D = data.shape
        if D != self.feat_dim:
            raise ValueError(
                f"Training data expected to have {self.feat_dim} feature dim. "
                f"Found {D}"
            )
        if is_train:
            if T != self.encode_len + self.decode_len:
                raise ValueError(
                    f"Training data expected to have {self.encode_len + self.decode_len}"
                    f" length on axis 1. Found length {T}"
                )
            X = data[:, :self.encode_len, :]
            y = data[:, self.encode_len:, 0]
        else:
            # for inference
            if T < self.encode_len:
                raise ValueError(
                    f"Inference data length expected to be >= {self.encode_len}"
                    f" on axis 1. Found length {T}"
                )
            X = data[:, -self.encode_len:, :]
            y = None
        return X, y

    def fit(self, train_data, valid_data, max_epochs=250, verbose=1):
        train_X, train_y = self._get_X_and_y(train_data, is_train=True)
        if valid_data is not None:
            valid_X, valid_y = self._get_X_and_y(
                valid_data, is_train=True)
        else:
            valid_X, valid_y = None, None

        self.batch_size = min(train_X.shape[0] // 8, 256)
        print(f"batch_size = {self.batch_size}")

        patience = get_patience_factor(train_X.shape[0])
        print(f"{patience=}")

        train_X, train_y = torch.FloatTensor(train_X), torch.FloatTensor(train_y)
        train_dataset = CustomDataset(train_X, train_y)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=int(self.batch_size), shuffle=True)
        
        if valid_X is not None and valid_y is not None:
            valid_X, valid_y = torch.FloatTensor(valid_X), torch.FloatTensor(valid_y)
            valid_dataset = CustomDataset(valid_X, valid_y)
            valid_loader = DataLoader(
                dataset=valid_dataset, batch_size=int(self.batch_size),  shuffle=True)
        else:
            valid_loader = None

        losses = self._run_training(train_loader, valid_loader, max_epochs,
                           use_early_stopping=True, patience=patience,
                           verbose=verbose)
        return losses
    
    def _run_training(self, train_loader, valid_loader, max_epochs,
                      use_early_stopping=True, patience=10, verbose=1):
        
        best_loss = 1e7
        losses = []
        min_epochs = 10
        for epoch in range(max_epochs):
            self.net.train()
            for data in train_loader:
                X,  y = data[0].to(device), data[1].to(device)
                # Feed Forward
                preds = self.net(X)
                # Loss Calculation
                loss = self.criterion(y, preds)
                # Clear the gradient buffer (we don't want to accumulate gradients)
                self.optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Weight Update: w <-- w - lr * gradient
                self.optimizer.step()

            current_loss = loss.item()

            if use_early_stopping:
                # Early stopping
                if valid_loader is not None:
                    current_loss = get_loss(self.net, device, valid_loader, self.criterion)
                losses.append({"epoch": epoch, "loss": current_loss})
                if current_loss < best_loss:
                    trigger_times = 0
                    best_loss = current_loss
                else:
                    trigger_times += 1
                    if trigger_times >= patience and epoch >= min_epochs:
                        if verbose == 1:
                            print(f'Early stopping after {epoch=}!')
                        return losses
            else:
                losses.append({"epoch": epoch, "loss": current_loss})
            # Show progress
            if verbose == 1:
                if epoch % self.print_period == 0 or epoch == max_epochs-1:
                    print(f'Epoch: {epoch+1}/{max_epochs}, loss: {np.round(current_loss, 5)}')

        return losses


    def predict(self, data):
        X = self._get_X_and_y(data, is_train=False)[0]
        pred_X = torch.FloatTensor(X)
        # Initialize dataset and dataloader with only X
        pred_dataset = CustomDataset(pred_X)
        pred_loader = DataLoader(
            dataset=pred_dataset, batch_size=32, shuffle=False)

        all_preds = []
        for data in pred_loader:
            # Get X and send it to the device
            X = data.to(device)
            preds = self.net(X).detach().cpu().numpy()
            preds = preds[:, -self.decode_len:]
            all_preds.append(preds)

        preds = np.concatenate(all_preds, axis=0)
        preds = np.expand_dims(preds, axis=-1)
        return preds
    
    def evaluate(self, test_data):
        """Evaluate the model and return the loss and metrics"""
        x_test, y_test = self._get_X_and_y(test_data, is_train=True)
        if self.net is not None:
            x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)
            dataset = CustomDataset(x_test, y_test)
            data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
            current_loss = get_loss(self.net, device, data_loader, self.criterion)
            return current_loss


    def save(self, model_path):
        model_params = {
            "encode_len": self.encode_len, 
            "decode_len": self.decode_len, 
            "feat_dim": self.feat_dim, 
            "latent_dim": self.latent_dim,
            "activation": self.activation,
            "rnn_unit": self.rnn_unit,
            "n_rnnlayers": self.n_rnnlayers,
            "bidirectional": self.bidirectional,
        }
        joblib.dump(model_params, os.path.join(model_path, MODEL_PARAMS_FNAME))
        torch.save(self.net.state_dict(), os.path.join(model_path, MODEL_WTS_FNAME))


    @classmethod
    def load(cls, model_path):
        model_params = joblib.load(os.path.join(model_path, MODEL_PARAMS_FNAME))
        classifier = cls(**model_params)
        classifier.net.load_state_dict(torch.load( os.path.join(model_path, MODEL_WTS_FNAME)))        
        return classifier

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.MODEL_NAME}"


def train_predictor_model(
    train_data: np.ndarray,
    valid_data: np.ndarray,
    forecast_length: int,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the forecaster model.

    Args:
        train_data (np.ndarray): The train split from training data.
        valid_data (np.ndarray): The valid split of training data.
        forecast_length (int): Length of forecast window.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        encode_len=train_data.shape[1] - forecast_length,
        decode_len=forecast_length,
        feat_dim=train_data.shape[2],
        **hyperparameters,
    )
    model.fit(
        train_data=train_data,
        valid_data=valid_data,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: np.ndarray
) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (np.ndarray): The test input data for forecasting.

    Returns:
        np.ndarray: The forecast.
    """
    return model.predict(test_data)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)



if __name__ == "__main__":     
    
    N = 100
    T = 25
    D = 3
    
    model = Net(
        feat_dim=D,
        latent_dim=13,
        n_cnnlayers=2,
        decode_len=10,
        activation='relu',
    )
    model.to(device=device)
    
    
    X = torch.from_numpy(np.random.randn(N, T, D).astype(np.float32)).to(device)
    
    preds = model(X)
    print(preds.shape)