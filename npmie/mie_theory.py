import os
import numpy as np
from scipy import interpolate
from scipy.special import sph_jn, sph_yn, sph_jnyn, riccati_jn, riccati_yn
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.realpath(__file__))


def get_optical_constants(metal='Au'):
    fname = metal + 'JohnsonChristy.txt'
    optical_constants = np.genfromtxt(os.path.join(cwd, fname), names=True)
    return optical_constants


def get_surrounding_medium(name='air'):
    n_medium = {}
    n_medium['air'] = 1.0
    n_medium['water'] = 1.33
    n_medium['1.5'] = 1.5
    if name in n_medium:
        n = n_medium[name]
        return n
    elif type(name)==int or type(name)==float:
        n = name
        return n
    else:
        print 'medium not found'
        return -1


def interpolate_optical_constants(wavelength, optical_constants):
    # scipy interpolate values #
    f_n = interpolate.interp1d(optical_constants['wavelength'],\
                               optical_constants['n'], kind='cubic')
    f_k = interpolate.interp1d(optical_constants['wavelength'],\
                               optical_constants['k'], kind='cubic')
    n_interp = f_n(wavelength)
    k_interp = f_k(wavelength)

    c = 3e8; h = 6.626068e-34; e = 1.60217646e-19
    energy = (h*c/e) / (wavelength * 1e-9)

    # remake optical constants structured array #
    interpolated_constants = np.empty(wavelength.shape,
                                     dtype=[('energy', np.float32),\
                                            ('wavelength', np.float32),\
                                            ('n', np.float32),\
                                            ('k', np.float32)])
    interpolated_constants['energy'] = energy
    interpolated_constants['wavelength'] = wavelength
    interpolated_constants['n'] = n_interp
    interpolated_constants['k'] = k_interp
    return interpolated_constants


def convert_complex_constants(optical_constants):
    new_optical_constants = np.empty(optical_constants.shape,
                                     dtype=[('energy', np.float32),\
                                            ('wavelength', np.float32),\
                                            ('n_cmplx', np.complex64)])
    for row in range(optical_constants.shape[0]):
        eV, wl, n, k = optical_constants[row]
        new_optical_constants[row] = (eV, wl, n + 1j*k)
    return new_optical_constants


def sph_hn(n,x):
    # calculate spherical hankel, h(n,x) = j(n,x) + iy(n,x) #
    jn, djn, yn, dyn = sph_jnyn(n, x)
    hn = jn + 1j*yn
    dhn = djn + 1j*dyn
    return hn, dhn


def mie_coefficients(n_max,x,m):
    # calculate spherical bessels #
    jn, djn, yn, dyn= sph_jnyn(n_max, x)    # j(n,x), y(n,x)
    jm, djm, ym, dym = sph_jnyn(n_max, m*x) # j(n,mx), y(n,mx)
    # calculate spherical hankel #
    hn, dhn = sph_hn(n_max,x)               # h(n,x)

    # calculate riccati bessel functions #
    #psi_n, dpsi_n = riccati_jn(n_max,x)
    #psi_m, dpsi_m = riccati_jn(n_max,m*x)
    #chi_n, dchi_n = riccati_yn(n_max,x)
    #zeta_n = psi_n + 1j*chi_n
    #dzeta_n = dpsi_n + 1j*dchi_n
    #print dpsi_n

    # calculate riccati bessel functions 2 #
    dpsi_n = [x*jn[n-1] - n*jn[n] for n in range(0,len(jn))]
    dpsi_m = [m*x*jm[n-1] - n*jm[n] for n in range(0,len(jm))]
    dzeta_n = [x*hn[n-1] - n*hn[n] for n in range(0,len(hn))]
    #print dpsi_n

    a_n = ( m**2 * jm*dpsi_n - jn*dpsi_m ) / ( m**2 * jm*dzeta_n - hn*dpsi_m )
    b_n = ( jm*dpsi_n - jn*dpsi_m ) / ( jm*dzeta_n - hn*dpsi_m )
    return a_n, b_n


def mie_efficiencies(r, wavelength, n_sph, n_med):
    # calculate size parameter #
    x = n_med * (2*np.pi/wavelength) * r    # x = n_med*kr, size parameter
    m = n_sph/n_med
    n = 1                               # expansion order
    #n_max = int(np.ceil(x.real)+1)      # number of terms in series expansion
    n_max = int(x + 4*x**(1.0/3.0) + 2) # number of terms in series expansion

    Qscat = 0
    Qbscat = 0
    Qext = 0
    Qabs = 0
    a_n, b_n = mie_coefficients(n_max,x,m)
    a = 0; b = 0
    for n in range(1, n_max):
        a += a_n[n]
        b += b_n[n]
        Qscat += (2*n+1)*(abs(a_n[n])**2 + abs(b_n[n])**2)
        Qbscat += (2*n+1)*((-1)**n)*(abs(a_n[n])**2 + abs(b_n[n])**2)
        Qext += (2*n+1)*(a_n[n] + b_n[n]).real
    Qscat *= 2/x**2
    Qbscat *= 2/x**2
    Qext *= 2/x**2
    Qabs = Qext - Qscat
    return Qscat, Qbscat, Qext, Qabs


def mie_main(r, w1, w2, dw=1, material='Au', medium='air'):
    '''
    Calculates the mie scattering and extinction efficiency of spherical
    nanoparticles with radius R for wavelengths between w1 and w2.
    '''
    # define nanoparticle system #
    optical_constants = get_optical_constants(material)
    n_med = get_surrounding_medium(medium)
    # create wavelength scale #
    wavelength = np.arange(w1, w2, dw)
    # interpolate optical constants to match wavelength scale #
    optical_constants = interpolate_optical_constants(wavelength, optical_constants)
    # create complex refractive index from optical constants #
    optical_constants = convert_complex_constants(optical_constants)
    # calculate mie scattering coefficients #
    mie_scattering = np.array([])
    mie_backscattering = np.array([])
    mie_extinction = np.array([])
    mie_absorption = np.array([])
    for data in optical_constants:
        wl = data[1]
        n_sph = data[2]
        Qscat, Qbscat, Qext, Qabs = mie_efficiencies(r, wl, n_sph, n_med)
        #print Qscat, Qbscat, Qext, Qabs
        mie_scattering = np.append(mie_scattering, Qscat)
        mie_backscattering = np.append(mie_backscattering, Qbscat)
        mie_extinction = np.append(mie_extinction, Qext)
        mie_absorption = np.append(mie_absorption, Qabs)
    return wavelength, optical_constants['energy'], mie_scattering,\
           mie_backscattering, mie_extinction, mie_absorption


if __name__ == '__main__':
    diameter_np = float(raw_input('Enter nanoparticle diameter (nm): '))
    material = raw_input("Enter nanoparticle material: ")
    medium = raw_input("Enter surrounding medium: ")
    wavelength, energy, mie_scattering, mie_backscattering, mie_extinction,\
                mie_absorption = mie_main(diameter_np/2.0, 300, 1100, 1,\
                                          material, medium)
    fig = plt.figure()
    # wavelength plots #
    ax = fig.add_subplot(421)
    ax.plot(wavelength, mie_scattering, 'r', label='scattering')
    ax.set_xticklabels(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('scattering')
    ax = fig.add_subplot(423)
    ax.plot(wavelength, mie_backscattering, 'k', label='back-scattering')
    ax.set_xticklabels(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('back-scattering')
    ax = fig.add_subplot(425)
    ax.plot(wavelength, mie_extinction, 'b', label='extinction')
    ax.set_xticklabels(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('extinction')
    ax = fig.add_subplot(427)
    ax.plot(wavelength, mie_absorption, 'g', label='absorption')
    ax.set_ylabel('absorption')
    ax.set_xlabel('wavelength (nm)')
    # energy plots #
    ax = fig.add_subplot(422)
    ax.plot(energy, mie_scattering, 'r', label='scattering')
    ax.set_xticklabels(ax.get_xticklabels(), visible=False)
    ax.set_yticklabels(ax.get_yticklabels(), visible=False)
    ax = fig.add_subplot(424)
    ax.plot(energy, mie_backscattering, 'k', label='back-scattering')
    ax.set_xticklabels(ax.get_xticklabels(), visible=False)
    ax.set_yticklabels(ax.get_yticklabels(), visible=False)
    ax = fig.add_subplot(426)
    ax.plot(energy, mie_extinction, 'b', label='extinction')
    ax.set_xticklabels(ax.get_xticklabels(), visible=False)
    ax.set_yticklabels(ax.get_yticklabels(), visible=False)
    ax = fig.add_subplot(428)
    ax.plot(energy, mie_absorption, 'g', label='absorption')
    ax.set_yticklabels(ax.get_yticklabels(), visible=False)
    ax.set_xlabel('energy (eV)')
    #plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
