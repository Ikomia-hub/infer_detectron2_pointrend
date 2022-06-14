from infer_detectron2_pointrend import update_path
from ikomia import core, dataprocess
import copy
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from infer_detectron2_pointrend.PointRend_git.point_rend.config import add_pointrend_config
import os
import random

# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class PointRendParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.cuda = True
        self.proba = 0.8

    def setParamMap(self, param_map):
        self.cuda = int(param_map["cuda"])
        self.proba = int(param_map["proba"])

    def getParamMap(self):
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["proba"] = str(self.proba)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class PointRend(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)

        # Create parameters class
        if param is None:
            self.setParam(PointRendParam())
        else:
            self.setParam(copy.deepcopy(param))

        self.threshold = 0.5
        self.MODEL_NAME_CONFIG = "pointrend_rcnn_R_50_FPN_3x_coco"
        self.path_to_config = "/PointRend_git/configs/InstanceSegmentation/"+self.MODEL_NAME_CONFIG+".yaml"
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.cfg = get_cfg()
        add_pointrend_config(self.cfg)
        self.cfg.merge_from_file(self.folder + self.path_to_config)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.cfg.MODEL.WEIGHTS = os.path.join(self.folder, "models", "model_final_3c3198.pkl")
        self.loaded = False
        self.deviceFrom = ""

        # add output + set data type
        self.setOutputDataType(core.IODataType.IMAGE_LABEL, 0)
        self.addOutput(dataprocess.CImageIO(core.IODataType.IMAGE))
        self.addOutput(dataprocess.CGraphicsOutput())
        self.addOutput(dataprocess.CBlobMeasureIO())

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def run(self):
        self.beginTaskRun()

        # we use seed to keep the same color for our masks + boxes + labels (same random each time)
        random.seed(30)

        # Get input :
        img_input = self.getInput(0)
        src_image = img_input.getImage()

        # Get output :
        mask_output = self.getOutput(0)
        output_graph = self.getOutput(2)
        output_graph.setImageIndex(1)
        output_graph.setNewLayer("PointRend")
        output_measure = self.getOutput(3)

        # Get parameters :
        param = self.getParam()

        # predictor
        if not self.loaded:
            print("Chargement du modèle")
            if not param.cuda:
                self.cfg.MODEL.DEVICE = "cpu"
                self.deviceFrom = "cpu"
            else:
                self.deviceFrom = "gpu"
            self.loaded = True
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA check and load without CUDA 
        elif self.deviceFrom == "cpu" and param.cuda:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            add_pointrend_config(self.cfg)
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.merge_from_file(self.folder + self.path_to_config)
            self.cfg.MODEL.WEIGHTS = self.folder + self.path_to_model
            self.deviceFrom = "gpu"
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and not param.cuda:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            add_pointrend_config(self.cfg)
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.merge_from_file(self.folder + self.path_to_config)
            self.cfg.MODEL.WEIGHTS = self.folder + self.path_to_model
            self.deviceFrom = "cpu"
            self.predictor = DefaultPredictor(self.cfg)

        outputs = self.predictor(src_image)

        # get outputs instances
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        classes = outputs["instances"].pred_classes
        masks = outputs["instances"].pred_masks

        # to numpy
        if param.cuda:
            boxes_np = boxes.tensor.cpu().numpy()
            scores_np = scores.cpu().numpy()
            classes_np = classes.cpu().numpy()
        else :
            boxes_np = boxes.tensor.numpy()
            scores_np = scores.numpy()
            classes_np = classes.numpy()

        self.emitStepProgress()

        # keep only the results with proba > threshold
        scores_np_thresh = list()
        for s in scores_np:
            if float(s) > param.proba:
                scores_np_thresh.append(s)

        if len(scores_np_thresh) > 0:
            # create random color for masks + boxes + labels
            colors = [[0, 0, 0]]
            for i in range(len(scores_np_thresh)):
                colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255])

            # text labels with scores
            labels = None
            class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
            if classes is not None and class_names is not None and len(class_names) > 1:
                labels = [class_names[i] for i in classes]

            if scores_np_thresh is not None and labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores_np_thresh]

            # Show boxes + labels + data
            for i in range(len(scores_np_thresh)):
                box_x = float(boxes_np[i][0])
                box_y = float(boxes_np[i][1])
                box_w = float(boxes_np[i][2] - boxes_np[i][0])
                box_h = float(boxes_np[i][3] - boxes_np[i][1])
                # label
                prop_text = core.GraphicsTextProperty()
                # start with i+1 we don't use the first color dedicated for the label mask
                prop_text.color = colors[i+1]
                prop_text.font_size = 8
                prop_text.bold = True
                output_graph.addText("{} {:.0f}%".format(labels[i], scores_np_thresh[i]*100), box_x, box_y, prop_text)
                # box
                prop_rect = core.GraphicsRectProperty()
                prop_rect.pen_color = colors[i+1]
                prop_rect.category = labels[i]
                graphics_obj = output_graph.addRectangle(box_x, box_y, box_w, box_h, prop_rect)
                # object results
                results = []
                confidence_data = dataprocess.CObjectMeasure(dataprocess.CMeasure(core.MeasureId.CUSTOM, "Confidence"),
                                                             float(scores_np_thresh[i]),
                                                             graphics_obj.getId(),
                                                             labels[i])
                box_data = dataprocess.CObjectMeasure(dataprocess.CMeasure(core.MeasureId.BBOX),
                                                      [box_x, box_y, box_w, box_h],
                                                      graphics_obj.getId(),
                                                      labels[i])
                results.append(confidence_data)
                results.append(box_data)
                output_measure.addObjectMeasures(results)
            
            self.emitStepProgress()
            
            # label mask
            nb_objects = len(masks[:len(scores_np_thresh)])
            if nb_objects > 0:
                masks = masks[:nb_objects, :, :, None]
                mask_or = masks[0]*nb_objects
                for j in range(1, nb_objects):
                    mask_or = torch.max(mask_or, masks[j] * (nb_objects-j))
                mask_numpy = mask_or.byte().cpu().numpy()
                mask_output.setImage(mask_numpy)

                # output mask apply to our original image 
                # inverse colors to match boxes colors
                c = colors[1:]
                c = c[::-1]
                colors = [[0, 0, 0]]
                for col in c:
                    colors.append(col)
                self.setOutputColorMap(1, 0, colors)
        else:
            self.emitStepProgress()
        
        self.forwardInputImage(0, 1)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class PointRendFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_pointrend"
        self.info.shortDescription = "PointRend inference model of Detectron2 for instance segmentation."
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
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.path = "Plugins/Python/Segmentation"
        self.info.iconPath = "icons/detectron2.png"
        self.info.version = "1.1.0"
        self.info.keywords = "mask,rcnn,PointRend,facebook,detectron2,segmentation"

    def create(self, param=None):
        # Create process object
        return PointRend(self.info.name, param)
