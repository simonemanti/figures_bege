"""Functionality to decorate plotting functionality in updated defaults."""
from matplotlib import rcParams
import matplotlib as mpl
# Customized rc parameters for the c2db_2.0 paper
#mpl.use('pgf')
pgf_with_latex = {
        "pgf.texsystem": "pdflatex",
        "font.family":"sans-serif",
        "font.sans-serif":['FreeSans'],
        #"pgf.preamble": [
        #    r"\usepackage[utf8x]{inputenc}",
        #    r"\usepackage{tgheros}",
        #    r"\usepackage[T1]{fontenc}"
        #    ],
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.major.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.5,
        "lines.linewidth": 1.0,
        }

# Provide rcParams for imports
mpl.rcParams.update(pgf_with_latex)
# Provide columnwidth and textwidth for iop template
columnwidth = 3.13  # in inches
textwidth = 5.95  # in inches

def plotter(rcp=pgf_with_latex):
    """Decorator pattern for matplotlib plotters."""
    def decorator(plotter):
        return rcParamsDecorator(rcp, plotter)
    return decorator


class rcParamsDecorator:
    """Decorator class for matplotlib plotting methods."""
    def __init__(self, rcp, plotter):
        """Initiate rcParamsDecorator.

        Parameters
        ----------
        rcp : dict
            values to be updated from default rcParams
        plotter : function
            plotting method to decorate with custom rcParams
        """
        self._rcp = rcp
        self._plotter = plotter

    def __call__(self, *args, **kwargs):
        """Make plotter call with updated rcParams."""
        from matplotlib import rcParams
        rcParams.update(self._rcp)
        return self._plotter(*args, **kwargs)
