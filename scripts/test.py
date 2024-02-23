from pathlib import Path

import numpy as np
import scipy.ndimage
from absl import app, logging
from gco import pygco

from app.semantic_segmentation.datasets import SemanticSegmentation
from app.utils import read_image
from detectron2.data import Metadata, MetadataCatalog
from detectron2.utils.visualizer import VisImage, Visualizer

dataset_name: str = "pano_semseg_v4_ntuh_test"
result_name: str = "2024-01-04-075005"
image_name: str = "cate3_N054408_20210528_PX_16001001_1"

root_dir: Path = Path("/mnt/hdd/PANO")
data_dir: Path = Path(root_dir, "data")
result_dir: Path = Path(root_dir, "results", result_name)


def main(_):
    data_driver: SemanticSegmentation | None = SemanticSegmentation.register_by_name(
        dataset_name=dataset_name, root_dir=data_dir
    )
    if data_driver is None:
        raise ValueError(f"Dataset {dataset_name} not found.")

    metadata: Metadata = MetadataCatalog.get(dataset_name)

    #

    zoom: float = 0.5
    image_path: Path = Path(data_dir, "images", "NTUH", f"{image_name}.jpg")
    npz_path: Path = Path(result_dir, "inference", f"{image_name}.npz")

    visualize_dir: Path = Path(result_dir, "visualize")
    visualize_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path) as data:
        prob = data["prob"]

    num_classes, height, width = prob.shape
    image_rgb: np.ndarray = read_image(image_path)

    #

    prob = np.asarray(prob, dtype=np.float32) / (2**16)

    prob_resized = np.stack(
        [scipy.ndimage.zoom(p, zoom=zoom, order=2) for p in prob], axis=0
    )
    prob_transposed = np.moveaxis(prob_resized, 0, -1)

    # unary_cost = -np.log(np.maximum(prob_transposed, 1e-6))
    unary_cost = -prob_transposed
    # pairwise_cost = -np.eye(num_classes, dtype=np.float32)
    pairwise_cost = np.zeros((num_classes, num_classes), dtype=np.float32)

    n = np.arange(num_classes)
    x = np.where(np.isin((n - 1) // 8, [1, 2]), 1, -1) * ((n - 1) % 8 + 0.5)
    y = np.where(np.isin((n - 1) // 8, [0, 1]), 1, -1)

    pairwise_cost[0, 1:] = 1.0
    pairwise_cost[1:, 0] = 1.0
    pairwise_cost[1:, 1:] = np.sum(
        [
            0.8 * np.square((x[1:, None] - x[None, 1:]) / 15.0),
            0.2 * np.square((y[1:, None] - y[None, 1:]) / 2.0),
        ],
        axis=0,
    )

    semseg_mask = pygco.cut_grid_graph_simple(
        unary_cost=np.asarray(unary_cost, order="C"),
        pairwise_cost=np.asarray(pairwise_cost, order="C"),
        connect=8,
        n_iter=-1,
    )
    semseg_mask = np.reshape(semseg_mask, prob_resized.shape[1:])

    semseg_mask = scipy.ndimage.zoom(
        semseg_mask,
        zoom=(height / prob_resized.shape[1], width / prob_resized.shape[2]),
        order=0,
    )

    visualizer = Visualizer(image_rgb, metadata=metadata, scale=1.0)
    vis_image: VisImage = visualizer.draw_sem_seg(semseg_mask, alpha=0.5)

    vis_path: Path = Path(visualize_dir, f"{image_name}.gco.png")
    logging.info(f"Saving visualization to {vis_path}")
    vis_image.save(vis_path)


if __name__ == "__main__":
    app.run(main)
