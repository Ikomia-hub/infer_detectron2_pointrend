from ikomia import utils, core, dataprocess
from ikomia.utils import qtconversion
from infer_detectron2_pointrend.infer_detectron2_pointrend_process import PointRendParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class PointRendWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = PointRendParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # cuda parameter
        cuda_label = QLabel("Cuda")
        self.cuda_ckeck = QCheckBox()
        self.cuda_ckeck.setChecked(True)

        # conf_tresh parameter
        conf_tresh_label = QLabel("Threshold :")
       
        self.conf_tresh_spinbox = QDoubleSpinBox()
        self.conf_tresh_spinbox.setValue(0.8)
        self.conf_tresh_spinbox.setSingleStep(0.1)
        self.conf_tresh_spinbox.setMaximum(1)
        if self.parameters.conf_tresh != 0.8:
            self.conf_tresh_spinbox.setValue(self.parameters.conf_tresh)

        self.gridLayout.setColumnStretch(0,0)
        self.gridLayout.addWidget(self.cuda_ckeck, 0, 0)
        self.gridLayout.setColumnStretch(1,1)
        self.gridLayout.addWidget(cuda_label, 0, 1)
        self.gridLayout.addWidget(conf_tresh_label, 1, 0)
        self.gridLayout.addWidget(self.conf_tresh_spinbox, 1, 1)
        self.gridLayout.setColumnStretch(2,2)

        # Set widget layout
        layoutPtr = qtconversion.PyQtToQt(self.gridLayout)
        self.set_layout(layoutPtr)

        if not self.parameters.cuda:
            self.cuda_ckeck.setChecked(False)

    def on_apply(self):
        # Apply button clicked slot
        if self.cuda_ckeck.isChecked():
            self.parameters.cuda = True
        else:
            self.parameters.cuda = False
        self.parameters.conf_tresh = self.conf_tresh_spinbox.value()
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class PointRendWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_detectron2_pointrend"

    def create(self, param):
        # Create widget object
        return PointRendWidget(param, None)
