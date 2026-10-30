# Recipe: PointPillars / CenterPoint with OpenPCDet

**Not executed in this repo** (no GPU, and GitHub was unreachable from the build container).
Install steps for OpenPCDet change with CUDA and PyTorch versions. Follow the upstream
`docs/INSTALL.md` at a **pinned commit** and record it.

## 1. Environment (outline)

```bash
git clone https://github.com/open-mmlab/OpenPCDet && cd OpenPCDet && git checkout <pinned-sha>
# PyTorch matching your CUDA, then spconv for that CUDA (e.g. spconv-cu120)
pip install -r requirements.txt && python setup.py develop
model-lab env check   # confirm torch.cuda, spconv, pcdet detected
```

## 2. Data (KITTI 3D object, non-commercial licence)

Download from cvlibs.net after accepting the terms. Generate infos and the GT database with
OpenPCDet's `create_kitti_infos`. **Do not commit data or infos to this repo.**

## 3. Reproduce a model-zoo baseline first

```bash
cd tools
# evaluate the official checkpoint (download link in OpenPCDet README model zoo)
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../ckpts/pointpillar_7728.pth --batch_size 4
# then train from scratch, same config
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg_file cfgs/kitti_models/pointpillar.yaml
```

The checkpoint filename above is from prior knowledge, so verify it in the model-zoo table.
Success means your Car moderate AP_R40 is within about 1 point of the table for the same config.
Record the GPU count and batch size, because they affect results.

## 4. Adapting to a new LiDAR

Change one variable at a time and re-evaluate after each:
1. `POINT_CLOUD_RANGE` and z-range for the mounting height (check `model_lab.data.geometry` conventions against your sensor's frame).
2. `VOXEL_SIZE` (pillar grid must stay divisible by the backbone strides).
3. Intensity normalisation: different sensors report 0–1, 0–255, or reflectivity, so normalise explicitly.
4. Anchor sizes (PointPillars) / class groups (CenterPoint).

## 5. Failure modes checklist

- [ ] Boxes offset by h/2 in z → bottom-centre vs gravity-centre
- [ ] Boxes rotated 90° → KITTI camera `ry` not converted (`kitti_camera_boxes_to_lidar`)
- [ ] Good near, nothing beyond 50 m → range filter, voxel capacity overflow, or anchors
- [ ] Train/eval mismatch → different `max_points_per_voxel` / `MAX_NUMBER_OF_VOXELS` between train and test configs

## CenterPoint on nuScenes

Configs: `cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml` (name from prior
knowledge, so verify). Requires 10-sweep infos, and evaluation goes through `nuscenes-devkit`
(mAP and NDS).
