import os

CONFIG_NAME = None
CONFIG_PATH = "../conf"

##############################################################
# Provide (as env var), either:
# * 'SPLADE_CONFIG_NAME', this config in splade/conf will be used
# * or 'SPLADE_CONFIG_FULLPATH' (full path, from an exp, such as '/my/path/to/exp/config.yaml'

# if nothing is provided, 'config_default' is used
##############################################################

assert sum([v in os.environ.keys() for v in ["SPLADE_CONFIG_NAME", "SPLADE_CONFIG_FULLPATH"]]) <= 1

if "SPLADE_CONFIG_NAME" in os.environ.keys():
    CONFIG_NAME = os.environ["SPLADE_CONFIG_NAME"]
elif "SPLADE_CONFIG_FULLPATH" in os.environ.keys():
    CONFIG_FULLPATH = os.environ["SPLADE_CONFIG_FULLPATH"]
    CONFIG_PATH, CONFIG_NAME = os.path.split(CONFIG_FULLPATH)
else:
    CONFIG_NAME = "config_default"

if ".yaml" in CONFIG_NAME:
    CONFIG_NAME = CONFIG_NAME.split(".yaml")[0]
