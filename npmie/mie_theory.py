import os
import numpy as np
from scipy import interpolate
from scipy.special import sph_jn, sph_yn, sph_jnyn, riccati_jn, riccati_yn

cwd = os.path.dirname(os.path.realpath(__file__))
c = 3e8
h = 6.626068e-34
e = 1.60217646e-19


def get_optical_constants(metal='Au'):
    """
    Loads the requested optical constants from text files in the module directory.

    :rtype : object
    :param metal: set of optical constants to load (Au or Ag)
    :return:
    """
    fname = metal + 'JohnsonChristy.txt'
    optical_constants = np.genfromtxt(os.path.join(cwd, fname), names=True)
    return optical_constants


def interpolate_optical_constants(wavelength, optical_constants):
    """
    Interpolate the optical constants and return the complex refractive index for the
    requested wavelengths.

    :rtype : object
    :param wavelength:
    :param optical_constants:
    :return:
    """
    # interpolate optical constants to the given wavelengths #
    f_n = interpolate.interp1d(optical_constants['nm'],
                               optical_constants['n'], kind='cubic')
    f_k = interpolate.interp1d(optical_constants['nm'],
                               optical_constants['k'], kind='cubic')
    n_interp = f_n(wavelength)
    k_interp = f_k(wavelength)

    energy = (h*c/e) / (1e-9*wavelength)

    # remake optical constants structured array #
    interpolated_constants = np.empty(wavelength.shape,
                                      dtype=[('energy', np.float32),
                                             ('wavelength', np.float32),
                                             ('n_cmplx', np.complex64)])
    interpolated_constants['energy'] = energy
    interpolated_constants['wavelength'] = wavelength
    interpolated_constants['n_cmplx'] = n_interp + 1j*k_interp
    return interpolated_constants


def sph_hn(n,x):
    # calculate spherical hankel, h(n,x) = j(n,x) + iy(n,x) #
    jn, djn, yn, dyn = sph_jnyn(n, x)
    hn = jn + 1j*yn
    dhn = djn + 1j*dyn
    return hn, dhn


def calculate_mie_coefficients(n_max, x, m):
    """
    Calculates the Mie coefficients.

    :rtype : object
    :param n_max:
    :param x: size parameter
    :param m:
    """

    # calculate spherical bessels #
    jn, djn, yn, dyn= sph_jnyn(n_max, x)     # j(n,x), y(n,x)
    jm, djm, ym, dym = sph_jnyn(n_max, m*x)  # j(n,mx), y(n,mx)
    # calculate spherical hankel #
    hn, dhn = sph_hn(n_max, x)               # h(n,x)

    # calculate riccati bessel functions #
    #psi_n, dpsi_n = riccati_jn(n_max,x)
    #psi_m, dpsi_m = riccati_jn(n_max,m*x)
    #chi_n, dchi_n = riccati_yn(n_max,x)
    #zeta_n = psi_n + 1j*chi_n
    #dzeta_n = dpsi_n + 1j*dchi_n
    #print dpsi_n

    # calculate riccati bessel functions 2 #
    dpsi_n = [x*jn[n-1] - n*jn[n] for n in range(0, len(jn))]
    dpsi_m = [m*x*jm[n-1] - n*jm[n] for n in range(0, len(jm))]
    dzeta_n = [x*hn[n-1] - n*hn[n] for n in range(0, len(hn))]

    a_n = (m**2 * jm*dpsi_n - jn*dpsi_m) / (m**2 * jm*dzeta_n - hn*dpsi_m)
    b_n = (jm*dpsi_n - jn*dpsi_m) / (jm*dzeta_n - hn*dpsi_m)
    return a_n, b_n


def calculate_mie_efficiencies(r, wavelength, n_sph, n_med):
    """
    Calculates the mie efficiencies (q_scat, q_abs, q_ext, q_bscat) for a sphere in a
    dielectric medium at a given wavelength.

    :rtype : object
    :param r: radius of the sphere
    :param wavelength: wavelength of illumination
    :param n_sph: complex refractive index of the sphere
    :param n_med: real refractive index of the dielectric medium
    :return:
    """

    # calculate size parameter #
    x = n_med * (2*np.pi/wavelength) * r    # x = n_med*kr, size parameter
    m = n_sph/n_med
    n = 1                                # expansion order
    #n_max = int(np.ceil(x.real)+1)      # number of terms in series expansion
    n_max = int(x + 4*x**(1.0/3.0) + 2)  # number of terms in series expansion

    q_scat = 0
    q_bscat = 0
    q_ext = 0
    q_abs = 0
    a_n, b_n = calculate_mie_coefficients(n_max, x, m)
    a = 0
    b = 0
    for n in range(1, n_max):
        a += a_n[n]
        b += b_n[n]
        q_scat += (2*n+1)*(abs(a_n[n])**2 + abs(b_n[n])**2)
        q_bscat += (2*n+1)*((-1)**n)*(abs(a_n[n])**2 + abs(b_n[n])**2)
        q_ext += (2*n+1)*(a_n[n] + b_n[n]).real
    q_scat *= 2/x**2
    q_bscat *= 2/x**2
    q_ext *= 2/x**2
    q_abs = q_ext - q_scat
    return q_scat, q_bscat, q_ext, q_abs


def calculate_mie_spectra(wavelengths, r, material='Au', n_med=1.):
    """
    Calculates the mie scattering and extinction efficiency of spherical
    nanoparticles with radius r and given material surrounded by a medium n_med
    for a set of given wavelengths.
    :rtype : object
    :param wavelengths: array of wavelengths to calculate spectra from
    :param r: radius of the sphere
    :param material: material of the sphere (Au or Ag)
    :param n_med: refractive index of the surrounding dielectric medium
    """

    # define nanoparticle system #
    optical_constants = get_optical_constants(material)
    # interpolate optical constants to match wavelength scale #
    optical_constants = interpolate_optical_constants(wavelengths, optical_constants)
    # calculate mie scattering coefficients #
    mie_scattering = np.array([])
    mie_backscattering = np.array([])
    mie_extinction = np.array([])
    mie_absorption = np.array([])
    for data in optical_constants:
        wl = data[1]
        n_sph = data[2]
        q_scat, q_bscat, q_ext, q_abs = calculate_mie_efficiencies(r, wl, n_sph, n_med)
        mie_scattering = np.append(mie_scattering, q_scat)
        mie_backscattering = np.append(mie_backscattering, q_bscat)
        mie_extinction = np.append(mie_extinction, q_ext)
        mie_absorption = np.append(mie_absorption, q_abs)
    return mie_scattering, mie_backscattering, mie_extinction, mie_absorption


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    diameter_np = raw_input('Enter nanoparticle diameter (nm): ')
    material = raw_input("Enter nanoparticle material: ")
    medium = raw_input("Enter surrounding medium: ")
    if diameter_np == '': diameter_np = 80.
    else: diameter_np = float(diameter_np)
    if material == '': material = 'Au'
    if medium == '': medium = 1.
    else: medium = float(medium)
    wavelength = np.arange(300, 1200, 1)
    mie_scattering, mie_backscattering, mie_extinction,\
                mie_absorption = calculate_mie_spectra(wavelength, diameter_np/2.0,
                                          material, medium)
    fig = plt.figure()
    # wavelength plots #
    ax = fig.add_subplot(411)
    ax.plot(wavelength, mie_scattering, 'r', label='scattering')
    ax.set_xticklabels(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('scattering')
    ax = fig.add_subplot(412)
    ax.plot(wavelength, mie_backscattering, 'k', label='back-scattering')
    ax.set_xticklabels(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('back-scattering')
    ax = fig.add_subplot(413)
    ax.plot(wavelength, mie_extinction, 'b', label='extinction')
    ax.set_xticklabels(ax.get_xticklabels(), visible=False)
    ax.set_ylabel('extinction')
    ax = fig.add_subplot(414)
    ax.plot(wavelength, mie_absorption, 'g', label='absorption')
    ax.set_ylabel('absorption')
    ax.set_xlabel('wavelength (nm)')
    plt.tight_layout()
    plt.show()
