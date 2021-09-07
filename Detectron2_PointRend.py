from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class Detectron2_PointRend(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from Detectron2_PointRend.Detectron2_PointRend_process import Detectron2_PointRendProcessFactory
        # Instantiate process object
        return Detectron2_PointRendProcessFactory()

    def getWidgetFactory(self):
        from Detectron2_PointRend.Detectron2_PointRend_widget import Detectron2_PointRendWidgetFactory
        # Instantiate associated widget object
        return Detectron2_PointRendWidgetFactory()
