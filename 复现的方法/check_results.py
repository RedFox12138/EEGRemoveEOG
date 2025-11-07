import scipy.io as sio
import os

results_dir = 'results'
methods = ['SSA', 'VME_EFD', 'EWTICEEMDAN', 'ACMD']

for method in methods:
    metrics_path = os.path.join(results_dir, f'{method}_metrics.mat')
    pred_path = os.path.join(results_dir, f'{method}_predictions.mat')
    
    print(f"\n{'='*60}")
    print(f"{method}")
    print(f"{'='*60}")
    
    if os.path.exists(metrics_path):
        data = sio.loadmat(metrics_path)
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Metrics keys: {keys}")
        
        if 'metrics' in data:
            metrics = data['metrics']
            print(f"Metrics type: {type(metrics)}")
            print(f"Metrics shape: {metrics.shape if hasattr(metrics, 'shape') else 'N/A'}")
            
            # 尝试提取RRMSE_mean
            try:
                if metrics.dtype.names:
                    print(f"Struct fields: {metrics.dtype.names}")
                    if 'RRMSE_mean' in metrics.dtype.names:
                        rrmse = metrics['RRMSE_mean'][0][0]
                        print(f"RRMSE_mean: {rrmse}")
            except:
                pass
        
        # 直接查找RRMSE_mean
        if 'RRMSE_mean' in data:
            print(f"Direct RRMSE_mean: {data['RRMSE_mean']}")
    
    if os.path.exists(pred_path):
        pred_data = sio.loadmat(pred_path)
        pred_keys = [k for k in pred_data.keys() if not k.startswith('__')]
        print(f"Predictions keys: {pred_keys}")
        
        if 'predictions' in pred_data:
            pred = pred_data['predictions']
            print(f"Predictions shape: {pred.shape}")
            print(f"Predictions mean: {pred.mean():.4f}")
            print(f"Predictions std: {pred.std():.4f}")
