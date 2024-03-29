from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        from infer_detectron2_pointrend.infer_detectron2_pointrend_process import PointRendFactory
        # Instantiate process object
        return PointRendFactory()

    def get_widget_factory(self):
        from infer_detectron2_pointrend.infer_detectron2_pointrend_widget import PointRendWidgetFactory
        # Instantiate associated widget object
        return PointRendWidgetFactory()
