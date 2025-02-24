import cc_pipeline.config as config
from cc_pipeline.cldm.hack import disable_verbosity, enable_sliced_attention

disable_verbosity()

if config.save_memory:
    enable_sliced_attention()
