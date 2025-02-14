from vocab import Vocabulary
import evaluation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

RUN_PATH = "/home/shenxiang/sda/zhuliqi/MSCA/runs/runX/checkpoint/model_best.pth.tar"
DATA_PATH = "/home/shenxiang/sda/zhuliqi/MSCA/data/"

evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="testall",fold5=True)
