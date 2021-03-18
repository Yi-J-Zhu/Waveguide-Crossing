import os
import datetime
import meep as mp

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

'''
For each length, the function sweeps through a fine range of widths
and calculates the tranmittances

Arugments
    f: function that approximates optimal width in relation to length
    lengths: lengths to explore
    widths: array of with offsets for each length. I.e. the function will explore f(l) + widths   
'''
def sweep(f, lengths, width_range, wg_width=0.5, wg_height=0.25, resolution=20, crossing=False, save_img=False):
    
    # start housekeeping
    _out = 'wg_width:{}\nwg_height:{}\n\n'.format(wg_width, wg_height)
    # make a folder to store all output 
    date = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
    os.makedirs('out/{}'.format(date))
    file = open('out/{}/info.txt'.format(date),"w") 
    file.write('wg_width={}, wg_height={}, resolution={}, crossing={}\n'.format(wg_width, wg_height, resolution, crossing))
    file.close()
    # end housekeeping
    
    # materials
    n_sio2 = 1.4547
    sio2 = mp.Medium(epsilon = n_sio2**2)
    n_sin = 2.029 
    sin = mp.Medium(epsilon = n_sin**2)

    # source variables
    lam = 0.729
    fsrc = 1/lam
    kx = 2.3
    kpoint = mp.Vector3(kx)
    bnum = 1 # lowest band
    base_source = mp.GaussianSource(fsrc,fwidth=0.5*fsrc)

    symmetries = [mp.Mirror(mp.Y,phase=-1),mp.Mirror(mp.Z,phase=1)]
    pml_layers = [mp.PML(1.0)]

    # waveguide
    wg_w = wg_width
    wg_h = wg_height
    geometryWG = [mp.Block(center=mp.Vector3(),
                           size=mp.Vector3(mp.inf,wg_w,wg_h),
                           material=sin)]
    
    ### START SWEEP ###
    n_l = len(lengths)
    n_w = len(width_range)
    n = n_l*n_w
    _fluxes = np.zeros(n)
    _lengths = np.zeros(n)
    _widths = np.zeros(n)
    
    # replace with range if trange breaks
    for i in tqdm(range(n)):
        
        l = lengths[(int)(i/n_w)]
        w = f(l) + width_range[i%n_w]
                
        # MMI
        geometryMMI = []
        MMI_l = l
        MMI_w = w
        geometryMMI = [mp.Block(center=mp.Vector3(),
                                    size=mp.Vector3(MMI_l,MMI_w,wg_h),
                                    material=sin)]        

        # cell dimensions
        cell_x = MMI_l+4
        cell_y = cell_x
        cell_z = wg_h*10
        cell = mp.Vector3(cell_x, cell_y, cell_z)
        
        if(crossing):
            geometryWG += [mp.Block(center=mp.Vector3(),
                           size=mp.Vector3(wg_w,mp.inf,wg_h),
                           material=sin)]
            geometryMMI += [mp.Block(center=mp.Vector3(),
                                    size=mp.Vector3(MMI_w,MMI_l,wg_h),
                                    material=sin)] 

        sources = [mp.EigenModeSource(base_source,
                                          center = mp.Vector3(-cell_x/2.+1.1,0,0),
                                          size = mp.Vector3(0,wg_w*3.,wg_h*3.),
                                          direction=mp.NO_DIRECTION,
                                          eig_kpoint=kpoint,
                                          eig_band=bnum,
                                          eig_parity=mp.EVEN_Z+mp.ODD_Y,
                                          eig_match_freq=True,
                                          amplitude = 1/base_source.fourier_transform(fsrc))]  

        sim = mp.Simulation(cell_size=cell,
                            boundary_layers=pml_layers,
                            geometry=geometryWG + geometryMMI,
                            sources=sources,
                            symmetries=symmetries,
                            resolution=resolution)

        out = sim.add_flux(fsrc,0,1,mp.FluxRegion(center = mp.Vector3(cell_x/2.-1.1,0,0),
                                                       size = mp.Vector3(0,wg_w*3.,wg_h*3.)))
        sim.init_sim()

        if(save_img):
            plt.clf()
            sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(),size=mp.Vector3(cell.x,cell.y)))
            plt.savefig('out/{}/conf_l_{}_w_{}.png'.format(date,l,w),dpi=300,bbox_inches='tight')
            fig = plt.gcf()
            fig.set_size_inches(8,6)

        sim.run(until_after_sources=100)
        _fluxes[i] = mp.get_fluxes(out)[0]
        _lengths[i] = l
        _widths[i] = w
    ### END SWEEP ###
            
    # start housekeeping
    np.save('out/{}/flux.npy'.format(date), _fluxes)
    np.save('out/{}/length.npy'.format(date), _lengths)
    np.save('out/{}/width.npy'.format(date), _widths)
    # end housekeeping


from scipy.optimize import curve_fit

flux = np.load('out/2020-08-01 223928/flux.npy')
length = np.load('out/2020-08-01 223928/length.npy')
width = np.load('out/2020-08-01 223928/width.npy')

ws = []
ls = np.linspace(2,10,81)
for l in ls:
    m = np.max(flux[np.where(length==l)])
    w = width[np.where(np.logical_and(length==l, flux==m))]
    ws.append(w[0])

_l = np.array(ls[6:])
_w = np.array(ws[6:])


def f(l, a, b, c):
    return  a*l**2 + b*l + c
popt, pcov = curve_fit(f, _l, _w)

# np.linspace(2,10,161)
# np.linspace(-0.05, 0.05, 11)
# print(popt)

# sweep(lambda l: f(l, *popt), np.linspace(2,10,161), np.linspace(-0.05, 0.05, 11), resolution=20, crossing=False)
# sweep(lambda l: f(l, *popt), np.linspace(2,10,161), np.linspace(-0.05, 0.05, 11), resolution=20, crossing=True)
sweep(lambda l: f(l, *popt), [3], [0], resolution=20, crossing=True, )




