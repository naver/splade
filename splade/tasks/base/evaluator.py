import os

import torch

from ...utils.utils import restore_model


class Evaluator:
    def __init__(self, model, config=None, restore=True):
        """base class for model evaluation (inference)
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if restore:
             #### TO-DO: load adaper using adapter-hub load method #####
            if "adapter_name" in config.keys() and config["adapter_name"]:
                if hasattr(self.model, "module"): # data parallel
                    self.model.module.transformer_rep.transformer.load_adapter(os.path.join(config["checkpoint_dir"],f"model/{config['adapter_name']}_rep"))#f"{config['checkpoint_dir']}/model/{config["adapter_name"]}_rep")
                else:
                    self.model.transformer_rep.transformer.load_adapter(os.path.join(config["checkpoint_dir"],f"model/{config['adapter_name']}_rep"))#f"{config['checkpoint_dir']}/model/{config["adapter_name"]}_rep")
                    # load query adapter if it exists
                    if os.path.exists(os.path.join(config["checkpoint_dir"],f"model/{config['adapter_name']}_rep_q")):
                        self.model.transformer_rep_q.transformer.load_adapter(os.path.join(config['checkpoint_dir'], f"model/{config['adapter_name']}_rep_q"))
                self.model.eval()
                if torch.cuda.device_count() > 1:
                    print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
                    self.model = torch.nn.DataParallel(self.model)
                self.model.to(self.device)
            else:
                if self.device == torch.device("cuda"):
                    if "pretrained_no_yamlconfig" not in config or not config["pretrained_no_yamlconfig"]:
                        checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "model/model.tar"))
                        restore_model(model, checkpoint["model_state_dict"])
                        print(
                            "restore model on GPU at {}".format(os.path.join(config["checkpoint_dir"], "model/model.tar")))
                    self.model.eval()
                    if torch.cuda.device_count() > 1:
                        print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
                        self.model = torch.nn.DataParallel(self.model)
                    self.model.to(self.device)

                else:  # CPU
                    if "pretrained_no_yamlconfig" not in config or not config["pretrained_no_yamlconfig"]:
                        checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "model/model.tar"),
                                                map_location=self.device)
                        restore_model(model, checkpoint["model_state_dict"])
                        print(
                            "restore model on CPU at {}".format(os.path.join(config["checkpoint_dir"], "model/model.tar")))
        else:
            print("WARNING: init evaluator, NOT restoring the model, NOT placing on device")
        self.model.eval()  # => put in eval mode
