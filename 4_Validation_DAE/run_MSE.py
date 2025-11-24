import numpy as np
import pickle
import h5py

def main():
    with open("../../DAE_mse_sampleVSdata.pkl", "rb") as f:
        y_pred, Y_val_clean_norm, norm21, recon21 = pickle.load(f)

    recon_error_data = np.mean((norm21 - recon21) ** 2, axis=1)
    recon_error = np.mean((Y_val_clean_norm - y_pred) ** 2, axis=1)
    
    with h5py.File('data_mse.h5', 'w') as f:
        f.create_dataset('recon_error_data', data=recon_error_data)
        f.create_dataset('recon_error', data=recon_error)

if __name__ == '__main__':
    main()