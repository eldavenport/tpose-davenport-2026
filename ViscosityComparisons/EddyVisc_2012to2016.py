import sys
sys.path.append('../../')
from open_tpose import tpose2012to2016_kpp, tposeOct2012_hourly, tpose2012to2014_kpp
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
warnings.filterwarnings("ignore")
import xarray as xr

from dask.distributed import LocalCluster, Client

def main():
    cluster = LocalCluster()
    client = Client(cluster)
    plt.rcParams['font.size'] = 17
    prefix = ['diag_kpp','diag_state']

    ds = tpose2012to2016_kpp(prefix)

    lon = float(sys.argv[1])
    zMin = float(sys.argv[2])
    latMin = -0.1
    latMax = 0.1

    print('lon: ' + str(lon))
    print('lat min: ' + str(latMin))
    print('lat max: ' + str(latMax))

    lats = ds.YC.data
    lons = ds.XC.data
    depths = ds.Z.data

    latli = np.argmin(np.abs(lats - latMin))
    latui = np.argmin(np.abs(lats - latMax)) + 1
    lonidx = np.argmin(np.abs(lons - lon))
    zMax = 0
    depthli = np.argmin(np.abs(depths - zMax))
    depthui = np.argmin(np.abs(depths - zMin)) + 2

    N = len(ds.time)
    ds['time'] = range(len(ds.time))

    # ------------------------------------------------------------------------------------------------------------------------------------
    MAMli = 0
    MAMui = MAMli + 92
    JJAui = MAMui + 92
    SONui = JJAui + 92

    decorr_scale_days = 5 #days
    N_ind = (SONui-JJAui)/decorr_scale_days
    std_err_denom = np.sqrt(N_ind)

    print('2012')  
    SON_EddyVisc_2012 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time').compute()  
    SON_EddyVisc_2012_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time').compute()/std_err_denom  
    SON_EUC_Core_2012 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time').compute() 

    #2013
    DJFli = SONui 
    DJFui = DJFli + 31 + 31 + 28
    MAMui = DJFui + 31 + 30 + 31
    JJAui = MAMui + 30 + 31 + 31
    SONui = JJAui + 30 + 31 + 30

    decorr_scale_days = 10 #days
    N_ind = (DJFui)/decorr_scale_days
    std_err_denom = np.sqrt(N_ind)

    EddyVisc_2012 = ds.KPPviscA[:DJFui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time').compute()   
    EddyVisc_2012_std = ds.KPPviscA[:DJFui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time').compute()/std_err_denom

    decorr_scale_days = 5 #days
    N_ind = (SONui-JJAui)/decorr_scale_days
    std_err_denom = np.sqrt(N_ind)
 
    SON_EddyVisc_2013 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time').compute()   
    SON_EddyVisc_2013_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time').compute()/std_err_denom  
    SON_EUC_Core_2013 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time').compute() 

    #2014
    DJFli = SONui 
    DJFui = DJFli + 31 + 31 + 28
    MAMui = DJFui + 31 + 30 + 31
    JJAui = MAMui + 30 + 31 + 31
    SONui = JJAui + 30 + 31 + 30

 
    SON_EddyVisc_2014 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time').compute()   
    SON_EddyVisc_2014_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time').compute()/std_err_denom  
    SON_EUC_Core_2014 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time').compute() 

    #2015
    DJFli = SONui 
    DJFui = DJFli + 31 + 31 + 28
    MAMui = DJFui + 31 + 30 + 31
    JJAui = MAMui + 30 + 31 + 31
    SONui = JJAui + 30 + 31 + 30

    SON_EddyVisc_2015 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time').compute()   
    SON_EddyVisc_2015_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time').compute()/std_err_denom  
    SON_EUC_Core_2015 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time').compute() 

    #2016
    DJFli = SONui 
    DJFui = DJFli + 31 + 31 + 29 # leap year
    MAMui = DJFui + 31 + 30 + 31
    JJAui = MAMui + 30 + 31 + 31
    SONui = JJAui + 30 + 31 + 30

    SON_EddyVisc_2016 = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time').compute()   
    SON_EddyVisc_2016_std = ds.KPPviscA[JJAui:SONui,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time').compute()/std_err_denom  
    SON_EUC_Core_2016 = ds.Z[ds.UVEL[JJAui:SONui,:depthui,latli:latui,lonidx].mean(dim='YC').argmax(dim='Z')].mean(dim='time').compute() 

    error_indices = np.arange(0, len(SON_EddyVisc_2012), 4)
    ones = np.ones_like(SON_EddyVisc_2012)
    selective_errors = np.full_like(SON_EddyVisc_2012, np.nan)  # Initialize with NaN
    selective_errors[error_indices] = ones[error_indices]  # Place errors only at specified indices

    ds_hourly = tposeOct2012_hourly(prefix,ref_date='2012-10-01')

    N = len(ds_hourly.time)
    ds_hourly['time'] = range(0,N,1)

    DWLES_ds = loadmat('DanWhittLES_profiles.mat')
    DWLES_x = DWLES_ds['prof_eddy_visc_Dan'][10:,0]
    print(len(DWLES_x))
    DWLES_std = DWLES_ds['std_eddy_visc_Dan'][10:,0]
    DWLES_y = DWLES_ds['z'][10:,0]

    # estimated in matlab using dans output. need to check this.
    decorr_scale_DWLES = 930
    N_ind_DW = 8774/decorr_scale_DWLES
    std_err_denom_DW = np.sqrt(N_ind_DW)
    DWLES_std = DWLES_std/std_err_denom_DW

    N = N - 24
    decorr_scale_hours = 72
    N_ind = N/decorr_scale_hours
    std_err_denom = np.sqrt(N_ind)
    print(std_err_denom)

    Avg_EddyVisc = ds_hourly.KPPviscA[24:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').mean(dim='time').compute() # now this is 1xdepth
    Avg_EddyVisc_std = (ds_hourly.KPPviscA[24:,depthli:depthui,latli:latui,lonidx].mean(dim='YC').std(dim='time')).compute()/std_err_denom # now this is 1xdepth

    Dillon_89_x2 = np.array([1.8*10**-3, 6.5*10**-4, 7.0*10 **-4, 6.5*10**-4, 4.5*10**-4, 1.7*10**-4, 8.5*10**-5])
    Dillon_89_y2 = [-30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0]

    Dillon_89_x1 = np.array([1.8*10**-3, 5.5*10**-4, 3.5*10**-4, 9.0*10**-5, 4.5*10**-5, 2.0*10**-5])
    Dillon_89_y1 = [-32.0, -45.0, -55.0, -69.0, -80.0, -92.0]

    Pinkel_23_x1 = np.array([0.7, 0.45, 0.5, 0.3, 0.3, 0.5, 0.4, 0.25, 0.35, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1])*10**-2
    Pinkel_23_y1 = np.arange(-30,-201,-10)

    Pinkel_23_x2 = np.array([0.65, 0.5, 0.5, 0.45, 0.4, 0.45, 0.48, 0.5, 0.4, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.05, 0.0, 0.0])*10**-2
    Pinkel_23_y2 = Pinkel_23_y1

    QW_96_x1 = np.array([46, 40, 35, 30, 20, 17, 10, 8, 4, 4, 10, 20, 40])*10**-4
    QW_96_y1 = [0.0, -15.0, -25.0, -32.0, -45.0, -50.0, -67.0, -75.0, -100.0, -125.0, -145.0, -175.0, -200.0]
    QW_96_std = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,2,3,4,8])*10**-4

    WL88_y1 = np.array([2,5,10,15,20,25,27,30,45,55,65,75,85,95,103,112,125,133,145,155,168,175,187,192,200])*-1
    WL88_x1 = np.array([49,45,40,35,30,25,20,15,12,17,7,10,3,5,2,2,5,3,9,7,18,12,19,12,15])*10**-4


    BB_85_x1 = np.array([1.7*10**-3])
    BB_85_y1 = [-75.0]

    fig, ax = plt.subplots(figsize=(24,16),ncols=3)
    ax[0].plot(Pinkel_23_x1,Pinkel_23_y1,linestyle='--',marker='o',linewidth=2.0,color='mediumblue',label='P23a (+0.3)')
    ax[0].plot(Pinkel_23_x2,Pinkel_23_y2,linestyle='--',marker='o',linewidth=2.0,color='dodgerblue',label='P23b (+0.3)')
    ax[0].plot(QW_96_x1,QW_96_y1,marker='o',linewidth=2.0,markersize=8,color='r',label='QW96 (+0.4)')
    ax[0].fill_betweenx(QW_96_y1,QW_96_x1-QW_96_std,QW_96_x1+QW_96_std,color='r',label='_nolegend_',alpha=0.25)
    ax[0].plot(WL88_x1,WL88_y1,marker='o',linewidth=2.0,markersize=8,color='tab:purple',label='WL88 (-0.4)')
    ax[0].plot(EddyVisc_2012,ds.Z[depthli:depthui],linewidth=1.5, color='#17712c',label='TPOSE \'12 (-0.2)')
    ax[0].fill_betweenx(ds.Z[depthli:depthui],EddyVisc_2012-EddyVisc_2012_std,EddyVisc_2012+EddyVisc_2012_std,color='#17712c',label='_nolegend_',alpha=0.25)
    ax[0].plot(Avg_EddyVisc,ds_hourly.Z[depthli:depthui],linewidth=1.5,color='#0acd38',label='TPOSE OCT \'12 (+0.3)')
    ax[0].fill_betweenx(ds_hourly.Z[depthli:depthui],Avg_EddyVisc-Avg_EddyVisc_std,Avg_EddyVisc+Avg_EddyVisc_std,color='#0acd38',label='_nolegend_',alpha=0.25)
    ax[0].plot(DWLES_x,DWLES_y,linewidth=2.0,color='tab:orange',label='W22 (-0.3)')
    ax[0].fill_betweenx(DWLES_y,DWLES_x-DWLES_std,DWLES_x+DWLES_std,color='tab:orange',label='_nolegend_',alpha=0.25)
    ax[0].plot(Dillon_89_x2,Dillon_89_y2,marker='s',linewidth=1.5,markersize=8,color='k',label='D89 GPWOS (-0.75)')
    ax[0].plot(Dillon_89_x1,Dillon_89_y1,marker='o',linewidth=1.5,markersize=8,color='k',label='D89 MC (-0.75)')
    ax[0].plot(BB_85_x1,BB_85_y1,marker='*',color='m',label='BB85 (+0.1)',markersize=20)
    ax[0].semilogx()
    ax[0].axvline(0,linewidth=0.75,color='k',linestyle='--',label='_nolabel_')
    ax[0].set_title('TPOSE and Observations')
    ax[0].set_xlabel('$m^2/s$')
    ax[0].legend(loc='lower right',ncol=2,framealpha=1.0)
    ax[0].set_ylim(zMin,0)
    ax[0].set_ylabel('Z (m)')
    ax[0].set_xlim(5e-6,3.0*10**-2)
    ax[0].text(-0.12, 1.03, '(a)', transform=ax[0].transAxes, size=20, fontweight='bold')


    ax[1].plot(SON_EddyVisc_2012,ds.Z[depthli:depthui],linewidth=2.5,color='#808b96',label='SON 2012 (+0.3)')
    ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2012-SON_EddyVisc_2012_std,SON_EddyVisc_2012+SON_EddyVisc_2012_std,color='#808b96',label='_nolegend_',alpha=0.2)
    ax[1].axhline(SON_EUC_Core_2012,linewidth=1.5,color='#808b96',label='_nolabel_')
    ax[1].plot(SON_EddyVisc_2013,ds.Z[depthli:depthui],linestyle='--',linewidth=2.5,color='#808b96',label='SON 2013 (-0.2)')
    ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2013-SON_EddyVisc_2013_std,SON_EddyVisc_2013+SON_EddyVisc_2013_std,color='#808b96',label='_nolegend_',alpha=0.2)
    ax[1].axhline(SON_EUC_Core_2013,linestyle='--',linewidth=1.5,color='#808b96',label='_nolabel_')
    ax[1].plot(SON_EddyVisc_2014,ds.Z[depthli:depthui],linewidth=2.5,color='#c0392b',label='SON 2014 (+0.5)')
    ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2014-SON_EddyVisc_2014_std,SON_EddyVisc_2014+SON_EddyVisc_2014_std,color='#c0392b',label='_nolegend_',alpha=0.2)
    ax[1].axhline(SON_EUC_Core_2014,linewidth=1.5,color='#c0392b',label='_nolabel_')
    ax[1].plot(SON_EddyVisc_2015,ds.Z[depthli:depthui],linestyle='--',linewidth=2.5,color='#c0392b',label='SON 2015 (+2.4)')
    ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2015-SON_EddyVisc_2015_std,SON_EddyVisc_2015+SON_EddyVisc_2015_std,color='#c0392b',label='_nolegend_',alpha=0.2)
    ax[1].axhline(SON_EUC_Core_2015,linestyle='--',linewidth=1.5,color='#c0392b',label='_nolabel_')
    ax[1].plot(SON_EddyVisc_2016,ds.Z[depthli:depthui],linestyle='-.',linewidth=2.5,color='#082f99',label='SON 2016 (-0.7)')
    ax[1].fill_betweenx(ds.Z[depthli:depthui],SON_EddyVisc_2016-SON_EddyVisc_2016_std,SON_EddyVisc_2016+SON_EddyVisc_2016_std,color='#082f99',label='_nolegend_',alpha=0.2)
    ax[1].axhline(SON_EUC_Core_2016,linestyle='-.',linewidth=1.5,color='#082f99',label='_nolabel_')
    ax[1].semilogx()
    ax[1].set_ylim(zMin,0)
    ax[1].set_xlim(5e-6,3.0*10**-2)
    ax[1].legend(loc='lower right',framealpha=1.0)
    ax[1].set_xlabel('$m^2/s$')
    ax[1].set_title('TPOSE SON')
    ax[1].text(-0.12, 1.03, '(b)', transform=ax[1].transAxes, size=20, fontweight='bold')

    # Add monthly variability panel
    filename_surf = 'diag_kpp'
    filename_state = 'diag_state'

    zMax = 0
    latMin = -0.1
    latMax = 0.1
    prefix = [filename_surf,filename_state]
    ds = tpose2012to2014_kpp(prefix=prefix)

    ds['XC'] = ds.XC.astype(float)
    ds['YC'] = ds.YC.astype(float)
    ds['Z'] = ds.Z.astype(float)
    ds['XG'] = ds.XG.astype(float)
    ds['YG'] = ds.YG.astype(float)
    ds['Zl'] = ds.Zl.astype(float)

    # Montly means, mins, maxes
    KPP_viscA = ds.KPPviscA.sel(Zl=slice(zMax,zMin),YC=slice(latMin,latMax)).sel(XC=[lon],method='nearest').mean(dim='YC').squeeze()
    da_KPP_viscA = KPP_viscA.resample(time="MS").mean().compute()
    oct_KPP_viscA = da_KPP_viscA.sel(time=da_KPP_viscA.time.dt.month == 10).mean("time").compute()
    sep_KPP_viscA = da_KPP_viscA.sel(time=da_KPP_viscA.time.dt.month == 9).mean("time").compute()
    nov_KPP_viscA = da_KPP_viscA.sel(time=da_KPP_viscA.time.dt.month == 11).mean("time").compute()
    oct_KPP_min = da_KPP_viscA.sel(time=da_KPP_viscA.time.dt.month == 10).min("time").compute()
    sep_KPP_min = da_KPP_viscA.sel(time=da_KPP_viscA.time.dt.month == 9).min("time").compute()
    nov_KPP_min = da_KPP_viscA.sel(time=da_KPP_viscA.time.dt.month == 11).min("time").compute()
    oct_KPP_max = da_KPP_viscA.sel(time=da_KPP_viscA.time.dt.month == 10).max("time").compute()
    sep_KPP_max = da_KPP_viscA.sel(time=da_KPP_viscA.time.dt.month == 9).max("time").compute()
    nov_KPP_max = da_KPP_viscA.sel(time=da_KPP_viscA.time.dt.month == 11).max("time").compute()
    uvel = ds.UVEL.sel(Z=slice(zMax,zMin),YC=slice(latMin,latMax)).sel(XG=[lon],method='nearest').squeeze().mean(dim='YC').compute()

    # EUC Core calculation
    EUC_Core = ds.Z[uvel.argmax(dim='Z')]

    da_EUC_core = EUC_Core.resample(time="MS").mean()
    oct_EUC_core = da_EUC_core.sel(time=da_EUC_core.time.dt.month == 10).mean("time").compute()
    sep_EUC_core = da_EUC_core.sel(time=da_EUC_core.time.dt.month == 9).mean("time").compute()
    nov_EUC_core = da_EUC_core.sel(time=da_EUC_core.time.dt.month == 11).mean("time").compute()

    sep_KPP_viscA.plot(ax=ax[2],y='Zl',linewidth=2.5,color='tab:orange',label=r'Sep Mean',zorder=10)
    sep_KPP_min.plot(ax=ax[2],y='Zl',color='tab:orange',label='_nolabel_',zorder=10,linewidth=1.5,linestyle='--')
    sep_KPP_max.plot(ax=ax[2],y='Zl',color='tab:orange',label='_nolabel_',zorder=10,linewidth=1.5,linestyle='--')

    oct_KPP_viscA.plot(ax=ax[2],y='Zl',linewidth=2.5,color='tab:green',label=r'Oct Mean',zorder=10)
    oct_KPP_min.plot(ax=ax[2],y='Zl',color='tab:green',label='_nolabel_',zorder=10,linewidth=1.5,linestyle='--')
    oct_KPP_max.plot(ax=ax[2],y='Zl',color='tab:green',label='_nolabel_',zorder=10,linewidth=1.5,linestyle='--')

    nov_KPP_viscA.plot(ax=ax[2],y='Zl',linewidth=2.5,color='tab:blue',label=r'Nov Mean',zorder=10)
    nov_KPP_min.plot(ax=ax[2],y='Zl',color='tab:blue',label='_nolabel_',zorder=10,linewidth=1.5,linestyle='--')
    nov_KPP_max.plot(ax=ax[2],y='Zl',color='tab:blue',label='_nolabel_',zorder=10,linewidth=1.5,linestyle='--')

    ax[2].axhline(sep_EUC_core,linestyle='-',linewidth=1.5,color='tab:orange',label='Sep EUC Core')
    ax[2].axhline(oct_EUC_core,linestyle='-',linewidth=1.5,color='tab:green',label='Oct EUC Core')
    ax[2].axhline(nov_EUC_core,linestyle='-',linewidth=1.5,color='tab:blue',label='Nov EUC Core')

    ax[2].semilogx()
    ax[2].set_xlim(5e-6,3.0*10**-2)
    ax[2].set_ylim(zMin,0)
    ax[2].legend(loc='lower right',ncols=1)
    ax[2].set_title('Monthly Viscosity')
    ax[2].set_xlabel('$m^2/s$')
    ax[2].set_ylabel('Z (m)')
    ax[2].text(-0.12, 1.03, '(c)', transform=ax[2].transAxes,  size=20, fontweight='bold')

    plt.tight_layout()
    image_str = 'EddyVisc_Obs_InteranVar_MonthlyVar_PinkelStd.png'
    plt.savefig(image_str,format='png')
    plt.close()

    client.close()
    cluster.close()

if __name__ == "__main__":
    main()