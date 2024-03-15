import albumentations as albu
import albumentations.pytorch as apt

def get_training_augmentations(
                   m = [0,0,0], s = [1,1,1] 
    ):
    train_transform = [

        albu.Resize(height=200, width=200), 
        # albu.Resize(height=256, width=256), 

        albu.HorizontalFlip(p=0.5),
        # albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.GaussNoise(p=0.2),
        # albu.Perspective(p=0.5),

        albu.OneOf(
            [
                # bugs: clahe supports only uint8 inputs,不要
                # albu.CLAHE(p=1), 
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                # albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),

        albu.Normalize(mean = m, std = s),
        apt.ToTensorV2(),
    ]
    return albu.Compose(train_transform, additional_targets={'t2': 'image', 'mask3d': 'mask'}, is_check_shapes=False)

def get_validation_augmentations(
        m = [0,0,0], s = [1,1,1]):

        train_transform = [
          albu.Resize(height=200, width=200), 
        #   albu.Resize(height=256, width=256), 
          albu.Normalize(mean = m, std = s),
          apt.ToTensorV2(),
          ]

        # return albu.Compose(train_transform, additional_targets={'t2': 'image', 'mask3d': 'mask'})
        return albu.Compose(train_transform, additional_targets={'t2': 'image', 'mask3d': 'mask'}, is_check_shapes=False)

