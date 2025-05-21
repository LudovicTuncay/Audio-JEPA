from multiprocessing import Value
from src.data.components.audioset_dataset import collate_audioset_batch
import torch


class MaskCollator(object):

    def __init__(
        self,
        ratio=(0.4, 0.6),
        input_size=(224, 224),
        patch_size=(16, 16),
        npred=2,
    ):
        super(MaskCollator, self).__init__()
        
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size[0], input_size[1] // patch_size[1]
        self.ratio = ratio
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)
        
        collated_batch = collate_audioset_batch(batch)
        # collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        ratio = self.ratio
        ratio = ratio[0] + torch.rand(1, generator=g).item() * (ratio[1] - ratio[0])
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - ratio))
        
        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(B):
            m = torch.randperm(num_patches, generator=g)
            collated_masks_enc.append([m[:num_keep]])
            collated_masks_pred.append([m[num_keep:]])
            # pred_indices = m[num_keep:]
            # pred_masks = pred_indices.split(len(pred_indices) // 3)
            # collated_masks_pred.append(pred_masks)

        collated_batch.context_masks = torch.utils.data.default_collate(collated_masks_enc)
        collated_batch.prediction_masks = torch.utils.data.default_collate(collated_masks_pred)

        return collated_batch





if __name__ == "__main__":
    collator = MaskCollator(ratio=(0.8, 0.9))
    batch = [
        (torch.rand(1, 224, 224), torch.randint(0, 10, (1,))),
        (torch.rand(1, 224, 224), torch.randint(0, 10, (1,))),
    ]
    collated_batch, collated_masks_enc, collated_masks_pred = collator(batch)
    print("Infos")
    print(f"{len(batch)=}")
    print(f"{len(collated_batch)=}")
    print(f"{len(collated_masks_enc[0])=}")
    print(f"{len(collated_masks_pred[0])=}")
    print("1st element")
    print(f"{collated_masks_enc[0][0].shape=}")
    print(f"{collated_masks_pred[0][0].shape=}")
    print("2nd element")
    print(f"{collated_masks_enc[0][1].shape=}")
    print(f"{collated_masks_pred[0][1].shape=}")
    
    import matplotlib.pyplot as plt    
    # from utils import apply_masks
   
    # Plot masks
    plot_masks(collated_masks_enc[0][0], collated_masks_pred[0][0])
    
    