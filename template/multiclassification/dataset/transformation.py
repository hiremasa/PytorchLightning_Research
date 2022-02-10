from torchvision import transforms

class Transforms:
    def __init__(self, config) -> None:
        self._config = config

        self.train_transform = transforms.Compose(
            [
                transforms.CenterCrop((self._config.transform.crop_size, self._config.transform.crop_size)),
                transforms.RandomAffine(degrees=(-90, 90), shear=(-0.2, 0.2)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Resize(size=self._config.transform.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.CenterCrop((self._config.transform.crop_size, self._config.transform.crop_size)),
                transforms.Resize(size=self._config.transform.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
