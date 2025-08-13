import matplotlib.image as mpimg
import matplotlib as mpl
import os
import pkg_resources 


LOGO = pkg_resources.resource_filename(__name__,'newCerfacs.png')
ALPHA_VAL = 0.5


def add_credits(figure):
    """ 
    Add CERFACS credits to a matplotlib figure. 
    """

    # Specify the position and size of the logo

    # Adjust bottom margin
    figure.subplots_adjust(bottom=0.5)

    # Add Logo
    #logo_ax = figure.add_axes([0.8, 0.01, 0.18, 0.18], anchor='SE', zorder=-1)
    #logo_ax.imshow(LOGO, interpolation='bicubic', alpha=ALPHA_VAL)
    #logo_ax.axis('off')

    # Add Text
    figure.text(0.79, 0.01, "Provided by CERFACS/COOP Team\nhttps://cerfacs.fr/coop/satis",
                fontsize=10, color='black', ha='right', va='bottom', alpha=ALPHA_VAL)

    return figure

def tight_plot(figure):
    """
    Perform a tight_layout with custom parameters to leave some blank space for
    credits.
    Does not really work for now...
    
    :param figure: matplotlib.Figure 
    :type figure: matplotlib.Figure
    """

    figure.tight_layout()

    return figure
