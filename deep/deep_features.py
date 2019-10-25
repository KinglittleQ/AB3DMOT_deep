import copy
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion
from deep.data_classes import PointCloud, Box
from deep.NNmodel import Model
import time


class DeepFeatureModel:

    def __init__(self, model_path, kitti_path, n_points=2048, offset_BB=0.0, scale_BB=1.25):

        self.kitti_path = Path(kitti_path)
        self.calib_path = self.kitti_path / 'calib'
        self.velodyne_path = self.kitti_path / 'velodyne'

        self.n_points = n_points
        self.model = Model(bneck_size=128)
        self.model.load_chkpt(model_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print(self.device)

        self.model = self.model.to(self.device)

        self.offset_BB = offset_BB
        self.scale_BB = scale_BB


    def compute_feature(self, seq, frame, boxes):
        time1 = time.time()

        calib_path = self.calib_path / f"{seq}.txt"
        calib = self.read_calib_file(calib_path)
        calib = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))

        all_PC = self.getPC(seq, frame, calib)

        PCs = []
        for box in boxes:
            PCs.append(self.processPC(box, all_PC))  # [3, n_points]
        PCs = torch.stack(PCs, dim=0).to(self.device)  # [N, 3, n_points]

        time2 = time.time()

        features = self.model.encode(PCs)  # [N, 128]

        time3 = time.time()

        print(time2 - time1, time3 - time2)

        return features

    def compute_similarity(self, seq, frame, boxes1, boxes2):
        features1 = self.compute_feature(seq, frame, boxes1)
        features2 = self.compute_feature(seq, frame, boxes2)
        sim = F.cosine_similarity(features1, features2, dim=1)  # [N]
        sim = sim.detach().cpu().numpy()

        return sim

    @staticmethod
    def read_calib_file(filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data

    def getPC(self, seq, frame, calib):


        # read Point Cloud
        velodyne_path = str(self.velodyne_path / seq / f'{frame:06}.bin')
        PC = PointCloud(
            np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
        PC.transform(calib)

        return PC

    def processPC(self, box, PC):
        """
        box: [x, y, z, theta, l, w, h]
        """

        # coonstruct bbox
        # center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        # size = [box["width"], box["length"], box["height"]]
        center = [box[0], box[1], box[2]]
        size = [box[5], box[4], box[6]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box[3]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation)

        sample_PC = self.cropAndCenterPC(copy.deepcopy(PC), BB, offset=self.offset_BB, scale=self.scale_BB)
        sample_PC = self.regularizePC(sample_PC)

        return sample_PC

    def regularizePC(self, PC):
        PC = np.array(PC.points, dtype=np.float32)
        if np.shape(PC)[1] > 2:
            if PC.shape[0] > 3:
                PC = PC[0:3, :]
            if PC.shape[1] != self.n_points:
                new_pts_idx = np.random.randint(
                    low=0, high=PC.shape[1], size=self.n_points, dtype=np.int64)
                PC = PC[:, new_pts_idx]
            PC = PC.reshape(3, self.n_points)

        else:
            PC = np.zeros((3, self.n_points))

        return torch.from_numpy(PC).float()


    @staticmethod
    def cropPC(PC, box, offset=0, scale=1.0):
        box_tmp = copy.deepcopy(box)
        box_tmp.wlh = box_tmp.wlh * scale
        maxi = np.max(box_tmp.corners(), 1) + offset
        mini = np.min(box_tmp.corners(), 1) - offset

        x_filt_max = PC.points[0, :] < maxi[0]
        x_filt_min = PC.points[0, :] > mini[0]
        y_filt_max = PC.points[1, :] < maxi[1]
        y_filt_min = PC.points[1, :] > mini[1]
        z_filt_max = PC.points[2, :] < maxi[2]
        z_filt_min = PC.points[2, :] > mini[2]

        close = np.logical_and(x_filt_min, x_filt_max)
        close = np.logical_and(close, y_filt_min)
        close = np.logical_and(close, y_filt_max)
        close = np.logical_and(close, z_filt_min)
        close = np.logical_and(close, z_filt_max)

        new_PC = PointCloud(PC.points[:, close])
        return new_PC

    @staticmethod
    def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):

        new_PC = DeepFeatureModel.cropPC(PC, box, offset=2 * offset, scale=4 * scale)

        new_box = copy.deepcopy(box)

        rot_mat = np.transpose(new_box.rotation_matrix)
        trans = -new_box.center

        # align data
        new_PC.translate(trans)
        new_box.translate(trans)
        new_PC.rotate((rot_mat))
        new_box.rotate(Quaternion(matrix=(rot_mat)))

        # crop around box
        new_PC = DeepFeatureModel.cropPC(new_PC, new_box, offset=offset, scale=scale)

        if normalize:
            new_PC.normalize(box.wlh)
        return new_PC
