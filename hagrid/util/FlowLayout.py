

import io, base64
import matplotlib.pyplot as plt
from IPython.display import display_html, HTML, display

class FlowLayout(object):
    ''' A class / object to display plots in a horizontal / flow layout below a cell '''
    def __init__(self):
        # string buffer for the HTML: initially some CSS; images to be appended
        self.sHtml =  """
        <style>
        .floating-box {
        display: inline-block;
        margin: 1px;
        border: 3px solid #888888;  
        }
        </style>
        """
        
    def add_plot(self, oAxes, **dArgs):
        ''' Saves a PNG representation of a Matplotlib Axes object '''
        Bio=io.BytesIO() # bytes buffer for the plot
        fig = oAxes.get_figure()
        #fig.canvas.print_png(Bio, bbox_inches="tight") # make a png of the plot in the buffer
        #print(dArgs)
        #print(fig.canvas)
        fig.canvas.print_png(Bio, **dArgs) # make a png of the plot in the buffer

        # encode the bytes as string using base 64 
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        #self.sHtml+= 
        sHtmlChart = (
            '<div class="floating-box">'+ 
            '<img src="data:image/png;base64,{}\n">'.format(sB64Img)+
            '</div>')
        #display(HTML(self.sHtml + sHtmlChart))
        self.sHtml += sHtmlChart

    def add_image(self, oImg, height=None):
        ''' Saves a JPEG representation of a PIL image '''
        if height is not None:
            oImg = oImg.resize((int(oImg.width / oImg.height * height), height))

        Bio=io.BytesIO() # bytes buffer for the plot
        oImg.save(Bio, format="JPEG")

        # encode the bytes as string using base 64 
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        #self.sHtml+= 
        sHtmlChart = (
            '<div class="floating-box">'+ 
            '<img src="data:image/png;base64,{}\n">'.format(sB64Img)+
            '</div>')
        #display(HTML(self.sHtml + sHtmlChart))
        self.sHtml += sHtmlChart


    def PassHtmlToCell(self):
        ''' Final step - display the accumulated HTML '''
        display(HTML(self.sHtml))
        