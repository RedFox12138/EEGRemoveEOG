import torch
import torch.optim as optim
import torch.nn.functional as F
import utils
import data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from model import DenoiseEEG


class TrainingConfig:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_configs = {
            'EMG': {'input_channels': 1, 'seq_len': 512, 'hidden_dim': 128},
            'EOG': {'input_channels': 1, 'seq_len': 512, 'hidden_dim': 128},
            'hybrid': {'input_channels': 1, 'seq_len': 512, 'hidden_dim': 128},
            'semi': {'input_channels': 19, 'seq_len': 200, 'hidden_dim': 128},
            'phy': {'input_channels': 1, 'seq_len': 256, 'hidden_dim': 128}
        }

def mask_input(data, mask_ratio=0.3):
    batch_size, num_channels, seq_len = data.shape
    mask = torch.rand(batch_size, num_channels, seq_len, device=data.device) < mask_ratio
    masked_data = data.clone()
    masked_data[mask] = 0  
    return masked_data, mask

def mae_loss(reconstructed, original, mask):
    reconstructed = reconstructed.to(mask.device)
    original = original.to(mask.device)
    time_loss = F.mse_loss(reconstructed, original, reduction='mean')
    return time_loss

class NoisyDataset1D(Dataset):
    def __init__(self, clean_eeg, cont_eeg, clean_targ=False):
        self.clean_targ = clean_targ
        self.signal = cont_eeg  
        self.clean_signal = clean_eeg  

    def __getitem__(self, index):
        noisy_signal = self.signal[index, :, :]
        clean_signal = self.clean_signal[index, :, :]
        source_signal = torch.tensor(noisy_signal, dtype=torch.float32)
        target_signal = torch.tensor(clean_signal, dtype=torch.float32)
        return source_signal, target_signal
    
    def __len__(self):
        return self.signal.shape[0]

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
    def train_epoch(self, model, train_dataloader, optimizer, mask_ratio):
        model.train()
        total_loss = 0
        
        for data, _ in train_dataloader:
            data = data.to(self.device)
            optimizer.zero_grad()
            
            masked_data, mask = mask_input(data, mask_ratio)
            exp_output = model(data)
            reconstructed = model(masked_data)
            
            loss = mae_loss(exp_output, reconstructed, mask) + F.mse_loss(exp_output, data)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        return total_loss / len(train_dataloader)
    
    def evaluate(self, model, dataloader):
        model.eval()
        total_loss = corr = snr_db = RRMSE_t = RRMSE_s = 0
        
        with torch.no_grad(): 
            for data, y in dataloader:
                x, y = data.to(self.device), y.to(self.device)
                reconstructed = model(x)

                loss = F.mse_loss(reconstructed, y)
                total_loss += loss.item()

                corr += utils.get_pearson_correlation(y, reconstructed)
                snr_db += utils.get_snr(y, reconstructed)
                RRMSE_t += utils.trrmse_metric(y, reconstructed)
                RRMSE_s += utils.frrmse_metric(y, reconstructed)
        
        num_batches = len(dataloader)
        return (total_loss / num_batches, corr / num_batches, 
                snr_db / num_batches, RRMSE_t / num_batches, RRMSE_s / num_batches)

def train(model, train_dataloader, val_dataloader, num_epochs=10, mask_ratio=0.3, lr=1e-3):
    config = TrainingConfig()
    trainer = ModelTrainer(config)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_metrics = {
        'rrmse_t': 1e9, 'rrmse_s': 1e9, 
        'snr': -1000000, 'cc': -100000
    }
    
    for epoch in range(num_epochs):

        avg_loss = trainer.train_epoch(model, train_dataloader, optimizer, mask_ratio)

        val_loss, corr, snr_db, RRMSE_t, RRMSE_s = trainer.evaluate(model, val_dataloader)
        
        if snr_db > best_metrics['snr']:
            best_metrics.update({
                'rrmse_t': RRMSE_t, 'rrmse_s': RRMSE_s,
                'snr': snr_db, 'cc': corr
            })

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, "
              f"Val loss: {val_loss:.4f}, RRMSE_t: {RRMSE_t:.3f}, "
              f"RRMSE_s: {RRMSE_s:.3f}, CC: {corr:.3f}, SNR: {snr_db:.3f}")
    
    print(f"Best SNR: {best_metrics['snr']:.3f}, Best CC: {best_metrics['cc']:.3f}, "
          f"Best RRMSE_t: {best_metrics['rrmse_t']:.3f}, Best RRMSE_s: {best_metrics['rrmse_s']:.3f}")

def create_model(dataset_type, config):
    model_config = config.dataset_configs.get(dataset_type)
    if not model_config:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return DenoiseEEG(
        model_config['input_channels'],
        model_config['seq_len'], 
        model_config['hidden_dim']
    ).to(config.device)

def prepare_data(dataset_type, snr_level):
    clean_all, noisy_all = data.getdata(dataset_type, snr_level)
    clean_train, clean_val_test, noisy_train, noisy_val_test = train_test_split(
        clean_all, noisy_all, test_size=0.2, random_state=42, shuffle=True
    )
    
    train_dataset = NoisyDataset1D(clean_train, noisy_train)  
    val_dataset = NoisyDataset1D(clean_val_test, noisy_val_test) 

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    config = TrainingConfig()
    
    DATASET_TYPE = 'phy' 
    snr_level = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
    
    train_dataloader, val_dataloader = prepare_data(DATASET_TYPE, snr_level)
    model = create_model(DATASET_TYPE, config)
    
    train(model, train_dataloader, val_dataloader, num_epochs=100, mask_ratio=0.1, lr=1e-3)