from infer_detectron2_pointrend import update_path
from ikomia import core, dataprocess, utils
import copy
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from infer_detectron2_pointrend.PointRend_git.point_rend.config import add_pointrend_config
import os
import random
import numpy as np


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class PointRendParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.cuda = True
        self.conf_tresh = 0.8

    def set_values(self, param_map):
        self.cuda = utils.strtobool(param_map["cuda"])
        self.conf_tresh = float(param_map["conf_tresh"])

    def get_values(self):
        param_map = {}
        param_map["cuda"] = str(self.cuda)
        param_map["conf_tresh"] = str(self.conf_tresh)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class PointRend(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)

        # Create parameters class
        if param is None:
            self.set_param_object(PointRendParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.threshold = 0.5
        self.MODEL_NAME_CONFIG = "pointrend_rcnn_R_50_FPN_3x_coco"
        self.path_to_config = "/PointRend_git/configs/InstanceSegmentation/" + self.MODEL_NAME_CONFIG + ".yaml"
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.cfg = get_cfg()
        add_pointrend_config(self.cfg)
        self.cfg.merge_from_file(self.folder + self.path_to_config)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/" \
                                 "164955410/model_final_edd263.pkl"
        self.loaded = False
        self.deviceFrom = ""
        self.predictor = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        self.begin_task_run()

        # we use seed to keep the same color for our masks + boxes + labels (same random each time)
        random.seed(30)

        # Get input :
        img_input = self.get_input(0)
        src_image = img_input.get_image()
        h, w, _ = np.shape(src_image)

        # Get parameters :
        param = self.get_param_object()

        # Set cache dir in the algorithm folder to simplify deployment
        os.environ["FVCORE_CACHE"] = os.path.join(os.path.dirname(__file__), "models")

        # predictor
        if not self.loaded:
            print("Loading model...")
            if not param.cuda or not torch.cuda.is_available():
                self.cfg.MODEL.DEVICE = "cpu"
                self.deviceFrom = "cpu"
            else:
                self.deviceFrom = "gpu"

            self.loaded = True
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA check and load without CUDA 
        elif self.deviceFrom == "cpu" and param.cuda and torch.cuda.is_available():
            print("Loading model...")
            self.cfg = get_cfg()
            add_pointrend_config(self.cfg)
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.merge_from_file(self.folder + self.path_to_config)
            self.cfg.MODEL.WEIGHTS = self.folder + self.path_to_model
            self.deviceFrom = "gpu"
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and not(param.cuda and torch.cuda.is_available()):
            print("Loading model...")
            self.cfg = get_cfg()
            add_pointrend_config(self.cfg)
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.merge_from_file(self.folder + self.path_to_config)
            self.cfg.MODEL.WEIGHTS = self.folder + self.path_to_model
            self.deviceFrom = "cpu"
            self.predictor = DefaultPredictor(self.cfg)

        outputs = self.predictor(src_image)
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
        self.set_names(class_names)

        # get outputs instances
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        classes = outputs["instances"].pred_classes
        masks = outputs["instances"].pred_masks
        self.emit_step_progress()

        # Show boxes + labels + data
        index = 0
        for box, score, cls, mask in zip(boxes, scores, classes, masks):
            if score > param.conf_tresh:
                x1, y1, x2, y2 = box.cpu().numpy()
                w = float(x2 - x1)
                h = float(y2 - y1)
                cls = int(cls.cpu().numpy())
                self.add_object(index, 0, cls, float(score), float(x1), float(y1), w, h, mask.byte().cpu().numpy())
            index += 1

        os.environ.pop("FVCORE_CACHE")

        self.emit_step_progress()
        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class PointRendFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_pointrend"
        self.info.short_description = "PointRend inference model of Detectron2 for instance segmentation."
        self.info.description = "PointRend inference model for instance segmentation trained on COCO. " \
                                "Implementation from Detectron2 (Facebook Research). " \
                                "PointRend algorithm provides more accurate segmentation mask. " \
                                "This plugin offers inference for ResNet50 backbone + FPN head."
        self.info.authors = "Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick"
        self.info.article = "PointRend: Image Segmentation as Rendering"
        self.info.journal = "ArXiv:1912.08193"
        self.info.year = 2019
        self.info.license = "Apache-2.0 License"
        self.info.repo = "https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend"
        self.info.documentation_link = "https://detectron2.readthedocs.io/index.html"
        self.info.path = "Plugins/Python/Segmentation"
        self.info.icon_path = "icons/detectron2.png"
        self.info.version = "1.5.0"
        self.info.keywords = "mask,rcnn,PointRend,facebook,detectron2,segmentation"

    def create(self, param=None):
        # Create process object
        return PointRend(self.info.name, param)
