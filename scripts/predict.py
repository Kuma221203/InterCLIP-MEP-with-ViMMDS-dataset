from pytorch_lightning import Trainer

from mmsd.lit_model import LitMMSDModel
from mmsd.dataset import MMSDDatasetModule

def infferent_predict(ckpt_path: str, path_clip_encoder: str = "openai/clip-vit-base-patch32") -> None:
    dataloader = MMSDDatasetModule(vision_ckpt_name = path_clip_encoder, text_ckpt_name = path_clip_encoder)
    model = LitMMSDModel.load_from_checkpoint(ckpt_path)
    trainer = Trainer()
    predict = trainer.predict(model, dataloader,ckpt_path=ckpt_path)
    print(predict)

if __name__ == "__main__":
    infferent_predict("./mmsd-results/tb_logs/t2v/version_0/checkpoints/mmsd-epoch=01-val_f1=0.84.ckpt")