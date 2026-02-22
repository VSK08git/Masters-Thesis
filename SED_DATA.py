# %%
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import pearsonr

# %%
#FERMI data
#light curve data
fermi = np.loadtxt("../../../media/varun-kelkar/DATA1/New-folder/LMU-MSc/Thesis/Mrk421/LC_FERMI_no_UL.txt")
fermi_mjd = fermi[:,0]
fermi_emjd = fermi[:,2]
fermi_flux = fermi[:,1]
fermi_eflux = fermi[:,3]
# Fermi SED data at low state
fermi_sed_ls = np.loadtxt("../../../media/varun-kelkar/DATA1/New-folder/LMU-MSc/Thesis/Mrk421/fermi_SED_points_lowstate.txt")
fermi_ls_en= fermi_sed_ls[:,1]
fermi_ls_en_low= fermi_sed_ls[:,0]
fermi_ls_en_high= fermi_sed_ls[:,2]
fermi_ls_nuFnu= fermi_sed_ls[:,4]
fermi_ls_nuFnu_low= fermi_sed_ls[:,3]
fermi_ls_nuFnu_high= fermi_sed_ls[:,5]
# Fermi SED data at mid state
fermi_sed_ms = np.loadtxt("../../../media/varun-kelkar/DATA1/New-folder/LMU-MSc/Thesis/Mrk421/fermi_SED_points_midstate.txt")
fermi_ms_en= fermi_sed_ms[:,1]
fermi_ms_en_low= fermi_sed_ms[:,0]
fermi_ms_en_high= fermi_sed_ms[:,2]
fermi_ms_nuFnu= fermi_sed_ms[:,4]
fermi_ms_nuFnu_low= fermi_sed_ms[:,3]
fermi_ms_nuFnu_high= fermi_sed_ms[:,5]

# %%
#fold gamma ray data in the IXPE range with beta parameter fixed
base_direct = "../../../media/varun-kelkar/DATA1/New-folder/LMU-MSc/Thesis/data_analysis/IXPE/IXPE_fold/fixed_beta/"
files_id = ["0430b", "0501b", "0502b", "0504b", "0505b", "0506b", "0510b", "0511b", "0512b", "0513b"]
days = ["0430", "0501", "0502", "0504", "0505", "0506", "0510", "0511", "0512", "0513"]
foldb_files = {}
SED_data = {} 
nu_data = {}
nuFnu_data = {}
EYhigh_data = {}
EYlow_data = {}
EXhigh_data = {}
EXlow_data = {}


for file in files_id:
    file_path = os.path.join(base_direct, f"fold_{file}.root")
    key = f"fold_{file}"
    foldb_files[key] = uproot.open(file_path)
    SED_data[key] = foldb_files[key]['deabsorbed_sed;1']

    nu_data[key] = SED_data[key].all_members['fX']
    nuFnu_data[key] = SED_data[key].all_members['fY']
    EYhigh_data[key] = SED_data[key].all_members['fEYhigh']
    EYlow_data[key] = SED_data[key].all_members['fEYlow']
    EXhigh_data[key] = SED_data[key].all_members['fEXhigh']
    EXlow_data[key] = SED_data[key].all_members['fEXlow']

# %%
#All radio data including polarization angle and degree
rad_data = pd.read_csv("../../../media/varun-kelkar/DATA1/New-folder/LMU-MSc/Thesis/Mrk421/mrk421_radio_data_all.csv", delimiter = ',')
rad_mjd = rad_data['mjd'].values
rad_nu = (rad_data['frequency(GHz)'].values)*1e9 #Hz
rad_flux = rad_data['flux'].values
rad_eflux = rad_data['error_flux'].values
rad_ePA = rad_data['error_PA'].values
rad_pa = rad_data['PA'].values
rad_PD = rad_data['PD'].values
rad_ePD = rad_data['error_PD'].values

# %%
h = 6.626e-34
c = 2.997e8
# To convert frequency to energy in GeV 
XRT_fact = h / 1.60218e-10
# To convert flux in Jy to ergs/cm^2/s^1/Hz
rad_fact = h*1e9*10e-23/1.602e-10
#
R_wl = 641.7e-9 # in metres central wavelength of R band
R_wl_err = 138e-9
V_wl = 545.8e-9
B_wl = 438.1e-9
I_wl = 798e-9
R_vega = 3.064e-20 #ergscm^-2 s^-1 Hz^-1
B_vega = 4.063e-20
I_vega = 2.416e-20
V_vega = 3.636e-20
R_nu = np.array([c/R_wl])
R_nu_err = np.array([c/R_wl_err])
V_nu = c/V_wl
I_nu = c/I_wl
B_nu = c/B_wl
#host flux 8 mJy subtract from R band
# error in optical flux calculated from magnitude
#f_obs = 10**(-0.4*mag)*f_vega
#err_f_obs = 10**(-0.4*mag)*f_vega*(0.4*err_mag)/1e-23

# %%
R_flux = np.array([12.927, 12.921, 12.922, 12.919, 12.952, 13.0075, 13.099, 13.062, 13.054, 12.999])
R_eflux = np.array([0.012, 0.012, 0.012, 0.014, 0.035, 0.0255, 0.017, 0.025, 0.012, 0.012])
R_MJD = np.array([60430.37351, 60431.36536, 60432.312975, 60434.2461, 60435.2363, 60436.2699, 60439.56894, 60440.54064, 60441.54868, 60443.52043])

# %%
#optical photometric data
opt_data = pd.read_csv('../../../media/varun-kelkar/DATA1/New-folder/LMU-MSc/Thesis/Mrk421/mrk421_photometry_data_all.csv', delimiter = ',')
opt_xy = opt_data['mjd'].values
opt_mag = opt_data['magnitude'].values
opt_emag = opt_data['error_magnitude'].values
band = opt_data['band'].values
tele = opt_data['telescope'].values
###########################
#F_vega = 3.064e-20 #erg cm-2 s-1 Hz-1

opt_mjd = []

for i in opt_xy:
    if i>= 2400000:
        opt_mjd.append(i - 2400000)
    else:
        opt_mjd.append(i)

opt_mjd = np.array(opt_mjd)

R_nuFnu = R_vega * R_nu *(10**(-0.4*(R_flux)))
R_nuFnu_err = R_nuFnu * 0.4 * np.log(10) * (R_eflux)
R_nuFnu_err

# %%
R_F = R_vega*(10**(-0.4*(R_flux)))
R_F_err = R_F * 0.4 * np.log(10) * (R_eflux)
R_F_err

# %%
###SWIFT-XRT SED data for everyday
folder_path = "../../../media/varun-kelkar/DATA1/New-folder/LMU-MSc/Thesis/SWIFT-XRT/SEDs/"

files = os.listdir(folder_path)
sed_data = {}

for file in files:

    num_part = file.split("_sed.dat")[0]

    var_name = f"sed_{num_part}"

    file_path = os.path.join(folder_path, file)

    data = np.loadtxt(file_path, delimiter=" " )
    globals()[var_name] = data.T
    #sed_data[var_name] = data.T

# %%
#seperating the optical photometric data according to the telescope and according to filter band
tele_names = ['LX200', 'Perkins', 'NOT', 'CAFOS', 'KANATA']
filter_bands = ['R', 'B', 'V', 'I']
opt_phot_data = {}

for tele_name in tele_names:
    opt_phot_data[tele_name] = {}  # Initialize dictionary for each telescope
    # Find indices where the telescope name matches
    tele_indices = np.where(tele == tele_name)[0]
    
    # loop will save the data from current telescope in loop in the following list
    current_tele_mag = opt_mag[tele_indices]
    current_tele_err_mag = opt_emag[tele_indices]
    current_tele_opt_mjd = opt_mjd[tele_indices]
    current_band_tele = band[tele_indices]

    for bands in filter_bands:
    # Find indices where the band is 'R' within the current telescope's data
        band_indices = np.where(current_band_tele == bands)[0]
        # Check if there are any data points for the current band
        if len(band_indices) > 0:
        # Extract the final filtered data
            opt_phot_data[tele_name][bands] = {
                f'{bands}_mag': current_tele_mag[band_indices],
                f'{bands}_err_mag': current_tele_err_mag[band_indices],
                f'{bands}_opt_mjd': current_tele_opt_mjd[band_indices]
            }


# %%
#photometric data averaging over a day
def group_mjds_and_flux(mjds, fluxes, errors):
    grouped_data = {}
    for mjd, flux, error in zip(mjds, fluxes, errors):
        day = int(mjd)
        if day not in grouped_data:
            grouped_data[day] = []
        grouped_data[day].append((mjd, flux, error))
    return grouped_data

def calculate_daily_averages(mjds, fluxes, errors):
    grouped_data = group_mjds_and_flux(mjds, fluxes, errors)
    mjd_average = []
    flux_average = []
    err_flux_average = []
    for day_data in grouped_data.values():
        mjd_values = [data[0] for data in day_data]
        flux_values = [data[1] for data in day_data]
        err_flux_values = [data[2] for data in day_data]
        mjd_average.append(np.mean(mjd_values))
        flux_average.append(np.mean(flux_values))
        err_flux_average.append(np.mean(err_flux_values))
    return np.array(mjd_average), np.array(flux_average), np.array(err_flux_average)

phot_dat_average = {}

for t in tele_names:
    if t in opt_phot_data:
        phot_dat_average[t] = {}  # new dictionary for the telescope inside original dictionary
        for b in filter_bands:
            if b in opt_phot_data[t]:
                data = opt_phot_data[t][b]
                mjd_key = f'{b}_opt_mjd'
                mag_key = f'{b}_mag'
                err_mag_key = f'{b}_err_mag'

                if mjd_key in data and mag_key in data and err_mag_key in data:
                    phot_mjds = data[mjd_key]
                    phot_fluxes = data[mag_key]
                    phot_errors = data[err_mag_key]
            
                    if len(phot_mjds) > 0:  # Check if there is data for the band
                        mjd_avg, flux_avg, err_flux_avg = calculate_daily_averages(phot_mjds, phot_fluxes, phot_errors)
                        phot_dat_average[t][b] = {
                            'mjd_average': mjd_avg,
                            'flux_average': flux_avg,
                            'err_flux_average': err_flux_avg
                        }
                    else:
                        print(f"Warning: No data found for '{b}' band for telescope '{t}'. Skipping.")
                else:
                    print(f"Warning: Data keys ('{mjd_key}', '{mag_key}', '{err_mag_key}') not found for band '{b}' in telescope '{t}'. Skipping.")
            else:
                print(f"Warning: '{b}' band data keys not found in 'opt_phot_data' for '{t}'. Skipping.")
    else:
        print(f"Warning: Data not found in 'opt_phot_data' for '{t}'. Skipping.")

        
####################################################################################################

# %%
sed_IXPE = [sed_00031540209, sed_00031540210, sed_00031540211, sed_00031540212, sed_00031540213, sed_00031540215, sed_00031540218, sed_00031540219, sed_00031540220, sed_00031540221]

# %%
ls_nu_dict = {}
for i, j, d in zip(sed_IXPE[:7], list(nu_data.values())[:7], days[:7]):
    ls_nu_dict[f"nu_{d}"] = np.concatenate([rad_nu, R_nu, i[0].flatten(), fermi_ls_en*(1.602e-3/6.626e-27), j*(1.602e-3/6.626e-27)])

# %%
ms_nu_dict = {}
for p, q, d in zip(sed_IXPE[6:], list(nu_data.values())[6:], days[6:]):
    ms_nu_dict[f"nu_{d}"] = np.concatenate([rad_nu, R_nu, p[0].flatten(), fermi_ms_en*(1.602e-3/6.626e-27), q*(1.602e-3/6.626e-27)])

# %%
ls_nu_err = {}
for i , j, d in zip(sed_IXPE[:7], list(EXhigh_data.values())[:7], days[:7]):
    ls_nu_err[f"nu_err_{d}"] = np.concatenate([np.zeros(np.shape(rad_nu)), R_nu_err, i[1].flatten(), np.zeros(np.shape(fermi_ls_en)), np.zeros(np.shape(j.flatten()))])

# %%
ms_nu_err = {}
for i , j, d in zip(sed_IXPE[6:], list(EXhigh_data.values())[6:], days[6:]):
    ms_nu_err[f"nu_err_{d}"] = np.concatenate([np.zeros(np.shape(rad_nu)), R_nu_err, i[1].flatten(), np.zeros(np.shape(fermi_ms_en)), np.zeros(np.shape(j.flatten()))])

# %%
ls_nuFnu_dict = {}
for a, b, d, x in zip(sed_IXPE[:7], list(nuFnu_data.values())[:7], days[:7], R_nuFnu[:7]):
    ls_nuFnu_dict[f"nuFnu_{d}"] = np.concatenate([rad_nu*rad_flux*1e-23, x.flatten() , a[2].flatten(), fermi_ls_nuFnu, b*1.60218])

# %%
ms_nuFnu_dict = {}
for e, f, d, x in zip(sed_IXPE[6:], list(nuFnu_data.values())[6:], days[6:], R_nuFnu[6:]):
    ms_nuFnu_dict[f"nuFnu_{d}"] = np.concatenate([rad_nu*rad_flux*1e-23, x.flatten() , e[2].flatten(), fermi_ms_nuFnu, f*1.60218])

# %%
ls_nuFnu_err = {}
for i , j, d, x in zip(sed_IXPE[:7], list(EYhigh_data.values())[:7], days[:7], R_nuFnu_err[:7]):
    ls_nuFnu_err[f"nuFnu_err_{d}"] = np.concatenate([rad_nu*rad_eflux*1e-23, x.flatten(), i[3].flatten(), fermi_ls_nuFnu_high, j*1.60218])

# %%
ms_nuFnu_err = {}
for i , j, d, x in zip(sed_IXPE[6:], list(EYhigh_data.values())[6:], days[6:], R_nuFnu_err[6:]):
    ms_nuFnu_err[f"nuFnu_err_{d}"] = np.concatenate([rad_nu*rad_eflux*1e-23, x.flatten(), i[3].flatten(), fermi_ms_nuFnu_high, j*1.60218])

# %%
nu_sed = {**ls_nu_dict, **ms_nu_dict}
nuFnu_sed = {**ls_nuFnu_dict, **ms_nuFnu_dict}
#nu_err_low_sed = {**ls_nu_err_low, **ms_nu_err_low}
#nu_err_high_sed = {**ls_nu_err_high, **ms_nu_err_high}
#nuFnu_err_low_sed = {**ls_nuFnu_err_low, **ms_nuFnu_err_low}
#nuFnu_err_high_sed = {**ls_nuFnu_err_high, **ms_nuFnu_err_high}
nuFnu_err_central = {**ls_nuFnu_err, **ms_nuFnu_err}
nu_err_central = {**ls_nu_err, **ms_nu_err}

# %%
all_dict = [nu_sed, nu_err_central, nuFnu_sed, nuFnu_err_central]

# %%
columns = ['nu', 'nu_err', 'nuFnu', 'nuFnu_err']

# %%
dict_values = []
day_data = []
for d in all_dict:
    dict_values.append(list(d.values()))

for i, day in enumerate(days):
    day_data = [d[i] for d in dict_values]

    # Trim to minimum length
    min_len = min(len(arr) for arr in day_data)
    if len(set(len(arr) for arr in day_data)) != 1:
        print(f"Trimming day {day} arrays to {min_len} entries due to mismatched lengths.")

    trimmed_data = [arr[:min_len] for arr in day_data]
    stacked_data = np.column_stack(trimmed_data)
    np.savetxt(f"SED_{day}.txt", stacked_data, header=" ".join(columns), delimiter = ',', fmt="%.6e")

print("All files saved (with trimming where needed).")


# %%
for i, day in enumerate(days):
    lengths = [len(d[i]) for d in dict_values]
    print(f"{day}: {lengths}")


