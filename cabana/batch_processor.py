"""Interactive (tkinter) wrapper around :class:`cabana.batch.BatchProcessor`.

The canonical batch-processing implementation lives in :mod:`cabana.batch`.
This module only adds two CLI-style behaviours on top of it:

* Pops tkinter file/folder dialogs in ``__init__`` to collect the parameter
  file, input folder, and output folder when the caller does not provide them
  programmatically.
* Adds an interactive ``check_running_status`` prompt so a user can resume
  from a checkpoint left by a previous run.
"""

import os
from pathlib import Path
from tkinter import Tk, filedialog

from .batch import BatchProcessor as _BatchProcessor
from .log import Log
from .utils import join_path


def _pick_paths_with_tkinter():
    """Pop tkinter dialogs to collect (param_file, input_folder, output_folder)."""
    gui = Tk()
    gui.withdraw()
    try:
        param_file = filedialog.askopenfilename(
            initialdir=os.path.expanduser("~/Documents/"),
            title="Choose Parameter File",
        )
        if not param_file or not os.path.exists(param_file):
            Log.logger.error("No path/folder has been selected. Abort!")
            os._exit(1)
        Log.logger.info(param_file + " has been selected.")

        input_folder = filedialog.askdirectory(
            initialdir=Path(param_file).parent.parent,
            title="Choose Input Directory",
        )
        if not input_folder or not os.listdir(input_folder):
            Log.logger.error("No path/folder has been selected. Abort.")
            os._exit(1)
        Log.logger.info(input_folder + " has been selected.")

        output_folder = filedialog.askdirectory(
            initialdir=os.path.dirname(input_folder),
            title="Choose Output Directory",
        )
        if not output_folder:
            Log.logger.error("No path/folder has been selected. Abort.")
            os._exit(1)
        Log.logger.info(output_folder + " has been selected.")
    finally:
        gui.destroy()
    return param_file, input_folder, output_folder


class BatchProcessor(_BatchProcessor):
    """Interactive batch processor — public API exported via ``cabana.__init__``.

    When called with no path arguments, pops tkinter dialogs to gather them.
    Otherwise behaves exactly like :class:`cabana.batch.BatchProcessor`.
    """

    def __init__(self, batch_size=5, param_file=None, input_folder=None,
                 output_folder=None, **kwargs):
        if param_file is None or input_folder is None or output_folder is None:
            param_file, input_folder, output_folder = _pick_paths_with_tkinter()
        super().__init__(param_file, input_folder, output_folder,
                         batch_size=batch_size, **kwargs)

    def check_running_status(self):
        """Interactively offer to resume from a checkpoint left by a prior run."""
        ckpt_path = join_path(self.output_folder, '.CheckPoint.txt')
        if not os.path.exists(ckpt_path):
            Log.logger.info("No checkpoint file found. Starting a new run.")
            self.resume = False
            return

        input_folder = ""
        batch_size = 5
        batch_num = 0
        ignore_large = False
        max_size = None
        save_visuals = None
        Log.logger.warning("A checkpoint file exists in the output folder.")
        with open(ckpt_path, "r") as f:
            for line in f:
                key, _, value = line.rstrip().partition(",")
                if key == "Input Folder":
                    input_folder = value
                elif key == "Batch Size":
                    batch_size = int(value)
                elif key == "Batch Number":
                    batch_num = int(value)
                elif key == "Ignore Large":
                    ignore_large = value.lower() == 'true'
                elif key == "Max Size":
                    max_size = int(value)
                elif key == "Save Visualisations":
                    save_visuals = value.lower() == 'true'

        self.resume = (os.path.exists(input_folder)
                       and os.path.samefile(input_folder, self.input_folder))

        for batch_idx in range(self.batch_num + 1):
            if not os.path.exists(join_path(self.output_folder, 'Batches',
                                            'batch_' + str(batch_idx))):
                Log.logger.warning('However, some necessary sub-folders are missing. '
                                   'A new run will start.')
                self.resume = False
                break

        while self.resume:
            user_input = input("Do you want to resume from last checkpoint? ([y]/n): ")
            if user_input.lower() in ("y", "yes"):
                Log.logger.info('Resuming from last check point.')
                self.batch_size = batch_size
                self.batch_num = batch_num
                self.ignore_large = ignore_large
                if max_size is not None:        # keep the splitting of the interrupted run
                    self.max_size = max_size
                    self.args.setdefault("Configs", {})["Max Size"] = max_size
                if save_visuals is not None:
                    self.save_visuals = save_visuals
                    self.args.setdefault("Configs", {})["Save Visualisations"] = save_visuals
                break
            elif user_input.lower() in ("n", "no"):
                Log.logger.info("Starting a new run.")
                self.resume = False
                break
            else:
                Log.logger.warning("Invalid input. Please enter y or n.")

    def run(self):
        self.check_running_status()
        super().run()


if __name__ == "__main__":
    import time
    from .log import setup_logging
    setup_logging()
    start_time = time.time()
    BatchProcessor(5).run()
    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    Log.logger.info("--- {:.0f} hours {:.0f} mins {:.0f} seconds ---".format(h, m, s))
    os._exit(0)
