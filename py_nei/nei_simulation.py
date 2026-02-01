"""
NEI simulation wrapper.

This module defines `NEISimulation`, a helper class to configure, run, and
visualize NEI simulations driven by the nei_post.exe executable.

Key capabilities:
- Manage simulation parameters and input files.
- Generate time-dependent history files.
- Execute the simulation executable.
- Parse and visualize outputs.

Author: Chengcai Shen
Update: 2026-02-01
"""

import glob
import os
import re
import matplotlib.pyplot as plt
import shutil
import subprocess
import logging
import numpy as np
from scipy.io import FortranFile
from typing import Optional, Dict, Any, List


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NEISimulation:
    """
    A class to manage the configuration, execution, and visualization of NEI simulations
    using the nei_post.exe executable.

    Attributes:
        exe_fname (str): Absolute path to the nei_post.exe executable.
        eigentable_path (str): Absolute path to the eigentable folders.
        working_dir (str): Directory where the simulation will be run and outputs saved.
        input_filename (str): Name of the input file required by the executable.
        parameters (Dict[str, Any]): Dictionary of parameters for the simulation.
    """

    def __init__(self,
                 exe_fname: Optional[str] = None, 
                 eigentable_dir: Optional[str] = None,
                 working_dir: Optional[str] = None):
        """
        Initialize the NEISimulation wrapper.

        Args:
            exe_fname (str, optional): Path to the nei_post.exe executable.
            eigentable_dir (str, optional): Path to the eigentable folders.
            working_dir (str, optional): Directory for simulation execution and file storage.
        """
        self.exe_fname = os.path.abspath(exe_fname) if exe_fname else None
        self.eigentable_dir = os.path.abspath(eigentable_dir) if eigentable_dir else None
        self.working_dir = os.path.abspath(working_dir) if working_dir else None
        self.parameters = {}
        self.temperature_density_history = {"time": [], "temperature": [], "density": []}
        
        self.kappa_folder_values = np.array([])  # To store available kappa values from eigentable_dir
        
        self._initialize_directories()
        self._initialize_eigentable_values()

    def configure(self, 
                  exe_fname: Optional[str] = None, 
                  eigentable_dir: Optional[str] = None,
                  working_dir: Optional[str] = None):
        """
        Configure or update the simulation paths after initialization.

        Args:
            exe_fname (str, optional): Path to the nei_post.exe executable.
            eigentable_dir (str, optional): Path to the eigentable folders.
            working_dir (str, optional): Directory for simulation execution and file storage.
        """
        if exe_fname:
            self.exe_fname = os.path.abspath(exe_fname)
        if eigentable_dir:
            self.eigentable_dir = os.path.abspath(eigentable_dir)
        if working_dir:
            self.working_dir = os.path.abspath(working_dir)
        self._initialize_directories()

    def _initialize_directories(self):
        """Helper to check and create directories."""
        if self.eigentable_dir and not os.path.isdir(self.eigentable_dir):
            logger.warning(f"Eigentable directory {self.eigentable_dir} does not exist.")
        
        if self.working_dir and not os.path.exists(self.working_dir):
            logger.warning(f"Working directory {self.working_dir} does not exist. Creating it.")
            os.makedirs(self.working_dir, exist_ok=True)
            
        if self.eigentable_dir is None:
            logger.warning("Eigentable directory is not set.")

    def _initialize_eigentable_values(self):
        if self.eigentable_dir:
            
            # Get the avatible kappa folders inside the eigentable_dir
            kappa_folders = [name for name in os.listdir(self.eigentable_dir)
                             if os.path.isdir(os.path.join(self.eigentable_dir, name)) and name.startswith('kappa_')]
            
            # Extract kappa values and store as np.array
            kappa_vals_list = []
            for k_name in kappa_folders:
                try:
                    # Assumes format "kappa_VALUE"
                    val_str = k_name.split('_')[1]
                    kappa_vals_list.append(float(val_str))
                except (IndexError, ValueError):
                    logger.warning(f"Skipping malformed kappa folder name: {k_name}")
            
            self.kappa_folder_values = np.array(sorted(kappa_vals_list))
            
            if kappa_folders:
                #logger.info(f"Available kappa folders found: {kappa_folders}")
                logger.info(f"Parsed kappa values: {self.kappa_folder_values}")
            else:
                logger.info("No kappa folders found; defaulting to Maxwellian tables.")
        else:
            raise ValueError("Eigentable directory is not set.\n"
                             "Please configure it before initializing eigentable values.")

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set or update the parameters required for the experiment.
        
        These parameters might be written to the input file before execution.

        Args:
            params (Dict[str, Any]): Key-value pairs of experiment parameters.
        """
        self.parameters.update(params)
        

    def set_history_data(self, time: Any, temperature: Any, density: Any) -> None:
        """
        Set the time-dependent temperature and density history.
        
        Args:
            time (array-like): Time points (1D).
            temperature (array-like): Temperature values (1D).
            density (array-like): Density values (1D).
            
        Raises:
            ValueError: If the input arrays do not have the same shape.
        """
        # Convert to numpy arrays to ensure consistency
        t_arr = np.array(time)
        te_arr = np.array(temperature)
        ne_arr = np.array(density)
        
        if not (t_arr.shape == te_arr.shape == ne_arr.shape):
            raise ValueError(f"Shape mismatch: time {t_arr.shape}, temperature {te_arr.shape}, density {ne_arr.shape} must be 1D and have the same length.")
            
        self.temperature_density_history["time"] = t_arr
        self.temperature_density_history["temperature"] = te_arr
        self.temperature_density_history["density"] = ne_arr
        
        logger.info(f"History data set with {len(t_arr)} points.")

    def write_input_file(self, index) -> str:
        """
        Generates the input file based on current parameters.

        This method should be customized to match the expected format of nei_exp
        
        Returns:
            str: Path to the generated input file.
            
        Raises:
            ValueError: If working_dir is not set.
        """
        if not self.working_dir:
            raise ValueError("Working directory is not set. Please call configure() or set working_dir.")

        file_path = os.path.join(self.working_dir, f"input_{index:05d}.txt")

        # Default values in the input.txt file for the Fortran program
        defaults = {
            'switch_eqi': 'T',
            'switch_log': 'T',
            'time_advance_mode': 3,
            'safety_factor_2': '1.0d-05',
            'safety_factor_3': 0.999,
            'num_elements': 14,
            'elements_list': '2,6,7,8,10,11,12,13,14,16,18,20,26,28',
            'ws_path_1': "'/path_to_eigentable_base/'",
            'ws_path_2': "'/path_to_te_ne_history.dat'",
            'ws_path_3': "'/path_to_initial_ionization_states.dat'",
            'ws_path_4': "'/path_to_output_folder/'",
            'output_basename': "'run_test'"
        }
        
        # Write the input file
        p = self.parameters
        
        try:
            with open(file_path, 'w') as f:
                f.write('"1. Set switch_eqi = T to start from ionization equilibrium:"\n')
                f.write(f"{p.get('switch_eqi', defaults['switch_eqi'])}\n")
                
                f.write('"2. Set switch_log = T to save logfiles:"\n')
                f.write(f"{p.get('switch_log', defaults['switch_log'])}\n")
                
                f.write('"3. Choose time-advance mode: 2 is explicit double-steps, and 3 is for Eigenvalue method"\n')
                f.write(f"{p.get('time_advance_mode', defaults['time_advance_mode'])}\n")
                
                f.write('"4. Set safety/accuracy factor in mode 2: default = 1.0e-5"\n')
                f.write(f"{p.get('safety_factor_2', defaults['safety_factor_2'])}\n")
                
                f.write('"5. Set safety/accuracy factor in mode 3: default = 0.999 "\n')
                f.write(f"{p.get('safety_factor_3', defaults['safety_factor_3'])}\n")
                
                f.write('"6. Set a number (and the list) of elements to be calculated:"\n')
                f.write(f"{p.get('num_elements', defaults['num_elements'])}\n")
                f.write(f"{p.get('elements_list', defaults['elements_list'])}\n")
                
                f.write('"7. Set workspace path: "\n')
                # 7.1. the eigentable path
                f.write(f"{p.get('ws_path_1', defaults['ws_path_1'])}\n")
                # 7.2. the te_ne_history path
                f.write(f"{p.get('ws_path_2', defaults['ws_path_2'])}\n")
                # 7.3. the initial ionization states path
                f.write(f"{p.get('ws_path_3', defaults['ws_path_3'])}\n")
                # 7.4. the output folder path
                f.write(f"{p.get('ws_path_4', defaults['ws_path_4'])}\n")

                f.write('"8. Set base-name of output files"\n')
                f.write(f"{p.get('output_basename', defaults['output_basename'])}\n")

            logger.info(f"Input file generated at: {file_path}")
        except IOError as e:
            logger.error(f"Failed to write input file: {e}")
            raise
        return file_path

    def write_history_files(self, index, 
                            time_select, temperature_select, density_select) -> None:
        """
        Generates the te_ne_history file required by the executable.

        The file will be saved in the working directory.
        
        Raises:
            ValueError: If working_dir is not set or history data is incomplete.
        """
        if not self.working_dir:
            raise ValueError("Working directory is not set. Please call configure() or set working_dir.")

        output_file = os.path.join(self.working_dir, "te_ne_history_{0:05d}.dat".format(index))
        ntime = len(temperature_select)
        with open(output_file, 'w') as f:
            f.write(f"{1}\n")  # only one trajactory
            f.write(f"{ntime}\n") # number of time steps
            f.write(' '.join(map(str, temperature_select)) + '\n')
            f.write(' '.join(map(str, density_select)) + '\n')
            f.write(' '.join(map(str, time_select)) + '\n')
            
    
    def generate_history_files(self) -> None:
        """
        Generate the input_index.txt and te_ne_history_index.dat files for the simulation.
        
        :param self: Description
        """
        # 
        # Check if the kappa_values parameter is a single value or a list
        #       
        kappa_values = self.parameters.get('kappa_values')
        if isinstance(kappa_values, (int, float, str)):
            #
            # if kappa is single value then generate input file directly for the whole 
            # history, else loop over kappa values to generate multiple input files
            
            # Find the proper kappa table based on self.parameters['kappa_values']                        
            table_str = self.func_kappavalue_to_tablestring(kappa_values)
            self.set_parameters({"ws_path_1": "'"+self.eigentable_dir + "/" + table_str + "/'"})
            self.set_parameters({"ws_path_2": "'"+self.working_dir + "/" + "te_ne_history_00000.dat'"})
            # ws_path_3 is defined by user already, no need to update for a simple run
            # ws_path_4 is defined by user already, no need to update for a simple run
            
            input_file_path = self.write_input_file(0)
            print(f"Input file generated at: {input_file_path}")
            
            self.write_history_files(0,
                                    self.temperature_density_history.get("time"),
                                    self.temperature_density_history.get("temperature"),
                                    self.temperature_density_history.get("density"))
            print("History files {:05d} generated.".format(0))
            
        else:
            #
            # kappa values change over time, need to generate multiple input 
            # files and history files
            #
            kappa_intents = np.array(kappa_values)
            ntime = len(kappa_intents)

            # Get the ik_index in the kappa_table list
            ik_index_list = np.zeros(ntime, dtype=int)
            for ik in range(ntime):
                ik_index_list[ik] = np.argmin(np.fabs(self.kappa_folder_values - kappa_intents[ik]))
        
            #
            # Set a series of input files and te_ne_history files for each kappa value
            #
            output_basename_original = self.parameters.get('output_basename', '')
            output_basename_original = output_basename_original.strip("'\"")
            index_history_patch = 0
            itime_0 = 0
            
            while itime_0 <= ntime - 2:
                idx_0 = ik_index_list[itime_0]
                # find the next index which is different from the current one
                itime_1 = itime_0 + 1
                while itime_1 < ntime - 1:
                    idx_1 = ik_index_list[itime_1]
                    if idx_1 != idx_0:
                        break
                    itime_1 += 1
                if itime_0 == ntime - 2:
                    itime_1 = ntime - 1
                    
                # now we have itime_0 and itime_1 for a segment
                if index_history_patch > 0:
                        self.set_parameters({"switch_eqi": 'F'})
                    
                kappa_intent_0 = kappa_intents[itime_0]
                table_str = self.func_kappavalue_to_tablestring(kappa_intent_0)

                self.set_parameters({"output_basename": "'"+output_basename_original + "_{0:05d}'".format(index_history_patch)})
                self.set_parameters({"ws_path_1": "'"+self.eigentable_dir + "/" + table_str + "/'"})
                self.set_parameters({"ws_path_2": "'"+self.working_dir + "/" + "te_ne_history_{0:05d}.dat'".format(index_history_patch)})
                self.set_parameters({"ws_path_3": "'"+self.working_dir + "/"+ output_basename_original + "_{0:05d}.dat'".format(index_history_patch-1)})
                
                input_file_path = self.write_input_file(index_history_patch)
                #print(f"Input file generated for kappa={kappa_intent_0} at: {input_file_path}")
                
                # select the history data for this segment
                time_select = self.temperature_density_history.get("time")[itime_0:itime_1+1]
                temperature_select = self.temperature_density_history.get("temperature")[itime_0:itime_1+1]
                density_select = self.temperature_density_history.get("density")[itime_0:itime_1+1]
                self.write_history_files(index_history_patch,
                                            time_select,
                                            temperature_select,
                                            density_select)
                #print(f"History files generated for the segment : {index_history_patch}.")
                
                # Update the history patch index
                index_history_patch += 1
                itime_0 = itime_1

                        

    def func_kappavalue_to_tablestring(self, kappa_value) -> str:
        """
        Converts a kappa value to the corresponding eigentable folder string.

        Args:
            kappa_value (float): The kappa value.

        Returns:
            str: The corresponding eigentable folder string.
        """
        if isinstance(kappa_value, str):
            table_str = 'Maxwellian'
        else:
            idx_kappa = np.argmin(np.fabs(self.kappa_folder_values - kappa_value))
            kappa_closest = self.kappa_folder_values[idx_kappa]
            logger.info(f"Using kappa value: {kappa_closest} for choosing the eigentable.")
            table_str = f'kappa_{kappa_closest:07.3f}'
        
        return table_str

    def run_simulation(self) -> int:
        """
        Executes the nei_post.exe program.
        
        It copies the executable to the working directory and runs it using mpiexe.

        Returns:
            int: The return code of the process (0 usually means success).

        Raises:
            ValueError: If executable path or working directory is not configured.
            FileNotFoundError: If the executable is not found.
            RuntimeError: If the execution fails.
        """
        if not self.exe_fname:
            raise ValueError("Executable path is not set. Please call configure() or set exe_path.")
        if not self.working_dir:
            raise ValueError("Working directory is not set. Please call configure() or set working_dir.")

        # Scan the working directory and find all input and history files
        input_files = sorted(glob.glob(os.path.join(self.working_dir, "input_*.txt")))
        history_files = sorted(glob.glob(os.path.join(self.working_dir, "te_ne_history_*.dat")))
        
        # check if the input files and history files match
        if len(input_files) != len(history_files):
            raise RuntimeError("Mismatch between number of input files and history files in working directory.")

        # check if the indices of input files are monotonically increasing
        indices = []
        for fname in input_files:
            match = re.search(r'input_(\d+)\.txt', os.path.basename(fname))
            if match:
                indices.append(int(match.group(1)))
        indices_sorted = sorted(indices)
        is_monotonic = all(x < y for x, y in zip(indices_sorted, indices_sorted[1:]))
        if not is_monotonic:
            raise RuntimeError("Input file indices are not monotonically increasing.")
        
        # Run simulations for each input/history file pair
        for index_patch in range(len(input_files)):
            print(f"Running simulation for patch index: {index_patch}")
            self.execute(input_file=input_files[index_patch])
        
        print("Simulation run completed.") 
            

    def execute(self, 
                input_file: Optional[str] = None) -> int:
        """
        Executes the compiled simulation executable in the working directory.

        This method invokes the executable (nei_post.exe) using mpirun. It assumes
        that the working directory is already prepared with the executable and 
        necessary input files.

        Returns:
            int: The return code of the process (0 indicates success, non-zero indicates failure).

        Raises:
            Exception: If the subprocess execution encounters a critical error.
        """
        # Define the command to run the executable
        # Always use the local executable name in the working directory
        command = ["mpirun", "-np", "1", self.exe_fname]
        if input_file:
            command.append(input_file)

        try:
            # Running the subprocess
            result = subprocess.run(
                command,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                check=False  # Set to True to raise exception on non-zero exit code immediately
            )

            if result.returncode == 0:
                logger.info("Experiment finished successfully.")
                logger.debug(f"Stdout: {result.stdout}")
            else:
                logger.error(f"Experiment failed with return code {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")

            return result.returncode

        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
            raise


    def visualize_evolution(self, logfile: Optional[str] = None):
        """
        Visualizes the logfile of the run.

        Args:
            logfile (str, optional): Path to the logfile to visualize. 
            If None, tries to determine from parameters.
        """        
        # Find all logfiles in the working directory if logfile is not provided
        if logfile is None:
            logger.info("Searching for logfiles in working directory: {:s}".format(self.working_dir))
            logfile_list = sorted(glob.glob(os.path.join(self.working_dir, '*.log')))
            
            if not logfile_list:
                # (Case 1) No logfile found
                logger.warning("No logfiles found in working directory. Skipping visualization.")
                return
            
            elif len(logfile_list) == 1:
                # (Case 2) Single logfile found
                logfile = logfile_list[0]
                
                # Read the single logfile
                nrow, ncol = 30, 30         # conce_ini shape 
                time_list = []
                conce_list = []

                with FortranFile(logfile, 'r') as f:
                    while True:
                        try:
                            time = f.read_reals(np.float64)[0]
                            conce_ini = f.read_reals(np.float64)
                            conce_ini = conce_ini.reshape((nrow, ncol), order='F')
                            time_list.append(time)
                            conce_list.append(conce_ini)
                        except Exception:
                            break  # the end of the file

                time_all = np.array(time_list)
                ntime = len(time_all)
                conce_all = np.zeros((30,30,ntime))
                for itime in range(ntime):
                    conce_all[:,:,itime] = conce_list[itime]
                print('time_all shape:', time_all.shape)
                print('conce_all shape:', conce_all.shape)
                
            else:
                # (Case 3) Multiple logfiles found - need to be combined
                nrow, ncol = 30, 30         # conce_ini shape 
                time_list = []
                conce_list = []
                n_logfiles = len(logfile_list)
                for ilog in range(n_logfiles):
                    logfile_ij = logfile_list[ilog]
                    #logger.info('Reading logfile: {}'.format(logfile_ij))
                    with FortranFile(logfile_ij, 'r') as f:
                        while True:
                            try:
                                time = f.read_reals(np.float64)[0]
                                conce_ini = f.read_reals(np.float64)
                                conce_ini = conce_ini.reshape((nrow, ncol), order='F')
                                time_list.append(time)
                                conce_list.append(conce_ini)
                            except Exception:
                                break  # the end of the file
                time_all = np.array(time_list)
                ntime = len(time_all)
                conce_all = np.zeros((30,30,ntime))
                for itime in range(ntime):
                    conce_all[:,:,itime] = conce_list[itime]
                print('time_all shape:', time_all.shape)
                print('conce_all shape:', conce_all.shape)

        
        # ----------------------------------------------------------------------
        # Plotting the results
        # ----------------------------------------------------------------------
        elem_stri = ['C', 'O', 'Fe']
        elem_list = [6, 8, 26]
        iion_start = [2, 2, 8]
        iion_end = [6, 8, 20]
        iion_step = [1, 1, 2]
        fig = plt.figure(figsize=(6, 10))

        for ipanel in range(len(elem_list)):
            natom = elem_list[ipanel]
            iatom = natom - 1
            
            ax = fig.add_subplot(len(elem_list), 1, 1+ipanel)
            ax.set_yscale('log')
            ax.set_ylim([1.0e-7, 1.0])
            ax.text(0.05, 0.925, '{0:s}'.format(elem_stri[ipanel]), transform=ax.transAxes)

            nstates = natom + 1
            colors = plt.cm.jet(np.linspace(0, 1, nstates))
            for iion in range(iion_start[ipanel], iion_end[ipanel]+1, iion_step[ipanel]):
                ax.plot(time_all, conce_all[iion, iatom, :], c=colors[iion], label='{0:s} {1:d}+'.format(elem_stri[ipanel], iion))
            
            ax.legend(ncol=3, loc=3)
            ax.set_ylabel('Ion Fraction')
            if (ipanel >= len(elem_list)-1):
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticklabels([])
            if (ipanel <= 0):
                ax.set_title('Charge States Evolution')

        plt.subplots_adjust(hspace=0.1)
        plt.savefig(self.working_dir+'/fig_evolution.png', dpi=200)  # or you can pass a Figure object to pdf.savefig
        plt.close()
        
        # Return to results
        return [time_all, conce_all]

if __name__ == "__main__":
    # Internal simple test
    pass
