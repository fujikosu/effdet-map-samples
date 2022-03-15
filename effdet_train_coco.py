# %%
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets.coco import CocoDetection


class EffDetCOCODataset(CocoDetection):
    def __init__(self, img_dir: Path, annotation_path: Path, transforms):
        super().__init__(root=str(img_dir), annFile=str(annotation_path))
        self.det_transforms = transforms

    def __getitem__(self, index):
        img, targets = super().__getitem__(index)
        img = np.array(img)
        bboxes = [target["bbox"] for target in targets]
        labels = [target["category_id"] for target in targets]
        transformed = self.det_transforms(image=img, bboxes=bboxes, labels=labels)
        transformed_img = transformed["image"]
        transformed_labels = transformed["labels"]

        _, new_h, new_w = transformed_img.shape

        # effdet needs yxyx coordinates, apply coordinate transformation
        bboxes_effdet = np.array(transformed["bboxes"])
        # x1,y1,w,h -> x1,y1,x2,y2
        bboxes_effdet[:, 2] += bboxes_effdet[:, 0]
        bboxes_effdet[:, 3] += bboxes_effdet[:, 1]
        # x1,y1,x2,y2 -> y1,x1,y2,x2
        bboxes_effdet[:, [0, 1, 2, 3]] = bboxes_effdet[:, [1, 0, 3, 2]]

        target = {
            "bboxes": torch.as_tensor(bboxes_effdet, dtype=torch.float32),
            "labels": torch.as_tensor(transformed_labels),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return transformed_img, target


# %%
import torchvision.transforms.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.utils import draw_bounding_boxes

img_dir = Path("odFridgeObjects/images")
annotation_dir = Path("odFridgeObjects/annotations")
id2label = {1: "can", 2: "carton", 3: "milk_bottle", 4: "water_bottle"}
dataset = EffDetCOCODataset(
    img_dir=img_dir,
    annotation_path=annotation_dir / "odFridgeObjects_coco_clean.json",
    transforms=ToTensorV2(),
)
img, target = dataset[9]
labels = [id2label[id] for id in target["labels"].tolist()]

img = draw_bounding_boxes(
    img, target["bboxes"][:, [1, 0, 3, 2]], labels, colors="Turquoise", width=2
)
img = F.to_pil_image(img.detach())
img


# %%

import albumentations as A


def get_train_transforms(img_size: int) -> A.Compose:
    """get data transformations for train set

    Args:
        img_size (int): image size to resize input data

    Returns:
        A.Compose: whole data transformations to apply
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=img_size, width=img_size),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_val_transforms(img_size: int):
    """get data transformations for val set

    Args:
        img_size (int): image size to resize input data

    Returns:
        A.Compose: whole data transformations to apply
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


# %%
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class EfficientDetDataModule(LightningDataModule):
    """LightningDataModule used for training EffDet
     This supports COCO dataset input

    Args:
        img_dir (Path): image directory
        annotation_dir (Path): annoation directory
        num_workers (int): number of workers to use for loading data
        batch_size (int): batch size
        img_size (int): image size to resize input data to during data
         augmentation
    """

    def __init__(
        self,
        img_dir: Path,
        annotation_dir: Path,
        num_workers: int,
        batch_size: int,
        img_size: int,
    ):
        super().__init__()
        self.train_transforms = get_train_transforms(img_size=img_size)
        self.val_transforms = get_val_transforms(img_size=img_size)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir

    def train_dataset(self) -> EffDetCOCODataset:
        return EffDetCOCODataset(
            img_dir=self.img_dir,
            annotation_path=self.annotation_dir / "odFridgeObjects_coco_clean.json",
            transforms=get_train_transforms(img_size=self.img_size),
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> EffDetCOCODataset:
        return EffDetCOCODataset(
            img_dir=self.img_dir,
            annotation_path=self.annotation_dir / "odFridgeObjects_coco_clean.json",
            transforms=get_val_transforms(img_size=self.img_size),
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset()
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return val_loader

    @staticmethod
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        images, targets = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets


# %%
from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet


def create_model(
    architecture: str, backbone: str, num_classes: int, img_size: int
) -> DetBenchTrain:
    """initialize a effdet model to train

    Args:
        architecture (str): the name of efficientdet architecture type to use
         the architecture needs to be supported by `effdet`
        backbone (str): the name of image featurizer (backbone) part to use
         supported featurizers are defined
         in timm (https://github.com/rwightman/pytorch-image-models#models)
         and can be examined with `timm.list_models()`
        num_classes (int): the number of classes
        img_size (int): image size to take in as input

    Returns:
        DetBenchTrain: model class with post processing embedded in
    """
    config = get_efficientdet_config(architecture)
    config.update({"num_classes": num_classes})
    config.update({"backbone_name": backbone})
    config.update({"image_size": (img_size, img_size)})

    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)


# %%
from pytorch_lightning import LightningModule
from torchmetrics.detection.map import MeanAveragePrecision


class EfficientDetModel(LightningModule):
    """LightningModule for efficientdet model

    Args:
        architecture (str): the name of efficientdet architecture type to use
         the architecture needs to be supported by `effdet`
        backbone (str): the name of image featurizer (backbone) part to use
         supported featurizers are defined
         in timm (https://github.com/rwightman/pytorch-image-models#models)
         and can be examined with `timm.list_models()`
        num_classes (int): the number of classes
        img_size (int): image size to take in as input
        learning_rate (float, optional): learning rate for training. Defaults to 0.02.
    """

    def __init__(
        self,
        architecture: str,
        backbone: str,
        num_classes: int,
        img_size: int,
        learning_rate: float = 0.02,
    ):
        super().__init__()
        self.model = create_model(
            architecture=architecture,
            backbone=backbone,
            num_classes=num_classes,
            img_size=img_size,
        )
        self.img_size = img_size
        self.lr = learning_rate
        self.num_classes = num_classes
        self.map = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        self.id2label = {1: "can", 2: "carton", 3: "milk_bottle", 4: "water_bottle"}

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _ = batch

        losses = self.model(images, annotations)

        self.log(
            "train_loss",
            losses["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_class_loss",
            losses["class_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_box_loss",
            losses["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        images, annotations, targets = batch
        outputs = self.model(images, annotations)

        detections = outputs["detections"]

        self.log(
            "val_loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_class_loss",
            outputs["class_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_box_loss",
            outputs["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        batch_size = images.shape[0]
        preds = []
        for i in range(batch_size):
            # detections: detection results in a tensor with shape [max_det_per_image, 6],
            #  each row representing [x_min, y_min, x_max, y_max, score, class]
            scores = detections[i, ..., 4]
            non_zero_indices = scores.nonzero()
            boxes = detections[i, non_zero_indices, 0:4]
            labels = detections[i, non_zero_indices, 5]
            # non_zero_indices retrieval adds extra dimension into dim=1
            #  so needs to squeeze it out
            preds.append(
                dict(
                    boxes=boxes.squeeze(dim=1),
                    scores=scores[non_zero_indices].squeeze(dim=1),
                    labels=labels.squeeze(dim=1),
                )
            )

        # target needs conversion from y1,x1,y2,x2 to x1,y1,x2,y2
        targets = []
        for i in range(batch_size):
            targets.append(
                dict(
                    boxes=annotations["bbox"][i][:, [1, 0, 3, 2]],
                    labels=annotations["cls"][i],
                )
            )
        self.map.update(preds=preds, target=targets)

    def validation_epoch_end(self, validation_step_outputs):
        mAPs = {"val_" + k: v for k, v in self.map.compute().items()}
        self.print(mAPs)
        mAPs_per_class = mAPs.pop("val_map_per_class")
        mARs_per_class = mAPs.pop("val_mar_100_per_class")
        self.log_dict(mAPs, sync_dist=True)
        self.log_dict(
            {
                f"val_map_{label}": value
                for label, value in zip(self.id2label.values(), mAPs_per_class)
            },
            sync_dist=True,
        )
        self.log_dict(
            {
                f"val_mar_100_{label}": value
                for label, value in zip(self.id2label.values(), mARs_per_class)
            },
            sync_dist=True,
        )
        self.map.reset()


# %%
from pytorch_lightning import Trainer

img_dir = Path("odFridgeObjects/images")
annotation_dir = Path("odFridgeObjects/annotations")
output_dir = Path("outputs")
img_size = 512
model_params = {
    "architecture": "efficientdet_d0",
    "backbone": "efficientnetv2_s",
    "num_classes": 4,
    "img_size": img_size,
}

datamodule = EfficientDetDataModule(
    img_dir=img_dir,
    annotation_dir=annotation_dir,
    num_workers=8,
    batch_size=8,
    img_size=img_size,
)

model = EfficientDetModel(**model_params)

trainer = Trainer(
    gpus=1,
    max_epochs=80,
    num_sanity_val_steps=1,
)
trainer.fit(model, datamodule)

# %%
