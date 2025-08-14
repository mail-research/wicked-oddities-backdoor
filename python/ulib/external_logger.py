import sys
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

class ExternalLogger(object):
    def __init__(self, args, run_name=None):
        self.logger_type = args.external_logger
        if args.external_logger == 'neptune':
            project = args.external_logger_args
            if not os.path.exists(os.path.expanduser('~/.neptune')):
                raise Exception('Please create .neptune file to store your credential!')
            api_token = open(os.path.expanduser('~/.neptune')).readline().strip()
            # import neptune.new as neptune
            import neptune
            self.run = neptune.init_run(
                project=project,
                 api_token=api_token,
                name=args.run_name
            ) # your credentials
            self.set_val('CMD', " ".join(sys.argv[:]))
        elif args.external_logger == 'wandb':
            api_token = open(os.path.expanduser('~/.wandb')).readline().strip()
            import wandb
            wandb.login(key=api_token)
            self.run = wandb.init(
                project=args.external_logger_args,
                name=args.run_name, 
                config=args
            )
        else:
            self.run = None

    def log_val(self, key, val, step=None):
        if self.logger_type == 'neptune':
            self.run[key].log(val, step=step)
        elif self.logger_type == 'wandb':
            self.run.log({key: val}, step=step)
    def set_val(self, key, val):
        if self.logger_type == 'neptune':
            self.run[key] = str(val)
    def log_img(self, key, img, step=None):
        if self.run is not None:
            from neptune.new.types import File
            if type(img) == str:
                self.run[key].log(File(img), step=step)
                
    def set_dict(self, d):
        if self.run is not None:
            for k, v in d.items():
                self.run[k] = str(v)
    def log_dict(self, d, step=None):
        if self.logger_type == 'neptune':
            for k, v in d.items():
                self.run[k].log(v, step=step)
        elif self.logger_type == 'wandb':
            self.run.log(d, step=step)
                
    def cleanup(self):
        if self.logger_type == 'neptune':
            self.run.stop()
        elif self.logger_type == 'wandb':
            self.run.finish()
