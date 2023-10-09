from pytorch_lightning.utilities.cli import LightningCLI

from msmer.datamodule import CROHMEDatamodule
from msmer.lit_msmer import LitMsMER

cli = LightningCLI(LitMsMER, CROHMEDatamodule)
