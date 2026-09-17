import tarfile
import numpy as np
from astropy.table import Table
from tqdm import tqdm

def sum_baryonic_xray_fluxes(tar_path="data/eRASS1_Main.v1.2.fits.tar.gz", output_path="data/erass_fluxes.fits"):
    """
    Extracts the FITS file from the eRASS1 tar.gz archive, extracts RA, Dec, 
    and band fluxes, sums them up, computes total errors via root-sum-squares,
    and saves the processed table to an output FITS file.
    """
    print(f"Opening archive: {tar_path}")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        fits_member = next((m for m in tar.getmembers() if m.name.endswith('.fits')), None)
        if not fits_member:
            raise FileNotFoundError("No .fits file found inside the tar.gz archive.")
        
        print(f"Reading FITS table from: {fits_member.name}")
        with tar.extractfile(fits_member) as f:
            with tqdm(total=fits_member.size, desc="Reading FITS HDU", unit="B", unit_scale=True) as pbar:
                table = Table.read(f, format='fits', hdu=1)
                pbar.update(fits_member.size)
                
    ra_col = 'RA'
    dec_col = 'DEC'
    
    ra = table[ra_col]
    dec = table[dec_col]
    
    flux_cols = [col for col in table.colnames if 'ML_FLUX' in col and 'ERR' not in col]
    error_cols = [fc.replace('ML_FLUX', 'ML_FLUX_ERR') for fc in flux_cols]
        
    print(f"Identified Flux columns: {flux_cols}")
    print(f"Identified Error columns: {error_cols}")
    
    valid_pairs = [(fc, ec) for fc, ec in zip(flux_cols, error_cols) if ec is not None]
    
    if not valid_pairs:
        raise ValueError("Could not automatically map flux columns to their respective error columns. Check table.colnames.")
        
    # Extract data across bands with a progress bar
    fluxes_list = []
    errors_list = []
    for fc, ec in tqdm(valid_pairs, desc="Extracting bands"):
        fluxes_list.append(table[fc].filled(0) if hasattr(table[fc], 'filled') else table[fc])
        errors_list.append(table[ec].filled(0) if hasattr(table[ec], 'filled') else table[ec])
        
    fluxes = np.array(fluxes_list, dtype=float)
    errors = np.array(errors_list, dtype=float)
    
    print("Summing fluxes and propagating errors via root-sum-squares...")
    total_flux = np.sum(fluxes, axis=0)
    total_error = np.sqrt(np.sum(errors**2, axis=0))
    
    output_table = Table()
    output_table['RA'] = ra
    output_table['Dec'] = dec
    output_table['Total_Flux'] = total_flux
    output_table['Total_Error'] = total_error
    
    print(f"Saving output table to {output_path}...")
    output_table.write(output_path, format='fits', overwrite=True)
    print("Processing complete.")
    
    return output_path

if __name__ == "__main__":
    sum_baryonic_xray_fluxes()