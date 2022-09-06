# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# speaker_reco_finetune.py 
#   --config-path=conf/ 
#   --config-name=ecapa_tdnn_finetune-cv.yaml 
#     model.train_ds.manifest_filepath=nemo_experiments/220823-Finetune-CV-validated/220825-Finetune-CV-eval/train-80-dlei-manifest.json 
#     model.train_ds.augmentor.speed.prob=0.5 
#     model.validation_ds.manifest_filepath=nemo_experiments/220823-Finetune-CV-validated/220825-Finetune-CV-eval/dev-80-dlei-manifest.json 
#     model.test_ds.manifest_filepath=nemo_experiments/220823-Finetune-CV-validated/220825-Finetune-CV-eval/eval1-within-labels-dlei-manifest.json 
#     trainer.max_epochs=5 
#     trainer.devices='[1]' 
#     model.decoder.num_classes=12 
#     model.train_ds.batch_size=32
#
#  Manifests look like:
#  { "audio_filepath": "", 
#    "offset": 0, 
#    "duration": 0, 
#    "label": "" }
#
#
#
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

seed_everything(42)


@hydra_runner(config_path="conf", config_name="titanet-finetune.yaml")
def main(cfg):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    _ = exp_manager(trainer, cfg.get("exp_manager", None))
    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    speaker_model.maybe_init_from_pretrained_checkpoint(cfg)
    trainer.fit(speaker_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator)
        if speaker_model.prepare_test(trainer):
            trainer.test(speaker_model)


if __name__ == '__main__':
    main()
