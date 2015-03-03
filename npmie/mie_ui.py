__author__ = 'alansanders'

import os

os.environ.setdefault('QT_API', 'pyside')

import numpy as np
from mie_theory import calculate_mie_spectra
from traits.api import HasTraits, Int, Float, Range, Array, Button, Instance, Property, Enum
from traits.api import cached_property, NO_COMPARE
from traitsui.api import View, Item, HGroup, VGroup
from matplotlib.figure import Figure
from traitsui_mpl_qt import MPLFigureEditor


class MieUI(HasTraits):
    """
    GUI interface for interaction with Mie calculator method
    """

    fig = Property(Instance(Figure))
    diameter = Range(10, 250, 80, mode='slider')
    np_material = Enum('Au', 'Ag')
    medium = Float(1.)
    update_button = Button('Update')
    spectra = Enum('scattering', 'absorption', 'extinction')

    wavelength_min = Float(400)
    wavelength_max = Float(1200)
    num_points = Int(1000)

    scattering = Array(dtype=np.float64, comparison_mode=NO_COMPARE)
    absorption = Array(dtype=np.float64, comparison_mode=NO_COMPARE)
    extinction = Array(dtype=np.float64, comparison_mode=NO_COMPARE)
    wavelength = Array(dtype=np.float32, comparison_mode=NO_COMPARE)

    view = View(
        VGroup(
            Item('fig', editor=MPLFigureEditor(), show_label=False),
            HGroup(
                'wavelength_min', 'wavelength_max', 'num_points', 'spectra',
            ),
            HGroup(
                Item('diameter', label='Diameter (nm)'),
                'np_material', 'medium',
                Item('update_button', show_label=False),
            ),
        ),
        resizable=True, title='NP Mie Scattering Calculator'
    )

    def __init__(self):
        self._init_fig()
        self.on_trait_change(self._update_wavelengths, ['wavelength_min', 'wavelength_max', 'num_points'])
        self.on_trait_change(self._update_button_fired, ['diameter', 'np_material', 'medium', 'wavelength'])
        self.on_trait_change(self._fig_changed, ['scattering', 'spectra'])
        self._update_wavelengths()

    @cached_property  # only draw the graph the first time it's needed
    def _get_fig(self):
        p = Figure()
        return p

    def _init_fig(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xlabel('wavelength (nm)')

    def _update_wavelengths(self):
        self.wavelength = np.linspace(self.wavelength_min, self.wavelength_max, self.num_points)

    def _update_button_fired(self):
        q_scat, q_bscat, q_ext, q_abs = calculate_mie_spectra(self.wavelength, self.diameter / 2.,
                                                              self.np_material, self.medium)
        self.scattering = q_scat
        self.absorption = q_abs
        self.extinction = q_ext

    def _fig_changed(self):
        if self.spectra == 'scattering':
            spectra = self.scattering
        elif self.spectra == 'absorption':
            spectra = self.absorption
        elif self.spectra == 'extinction':
            spectra = self.extinction
        ax = self.fig.axes[0]
        if not ax.lines:
            ax.plot(self.wavelength, spectra, 'r-')
        else:
            l = ax.lines[0]
            l.set_data(self.wavelength, spectra)
            ax.relim()
            ax.autoscale_view(True, True, True)
        self.fig.canvas.draw()


if __name__ == '__main__':
    MieUI().configure_traits()