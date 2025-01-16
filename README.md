# LLRAnalysis
These scripts run the analysis for the results of LLR method for Sp(4) lattice gauge theory using a modified version of HiRep, see https://github.com/dave452/Hirep_LLR_SP for the latest version, the Zenodo data release 10.5281/zenodo.13807993 for the version used in this study and https://github.com/claudiopica/HiRep for the main branch.
It can read the output file from HiRep and analyses it, this is then saved into CSV files.
These CSV files can be analysed and the results plotted.
The CSV files along with the scripts in this folder can be used to obtain the plots and results for the paper: Bennett, E., Lucini, B., Mason, D., Piai, M., Rinaldi, E., & Vadacchino, D. (2024). The density of states method for symplectic gauge theories at finite temperature. arXiv preprint arXiv:2409.19426.
-------------------------------------------------------------------------
## Creating Conda environment
To create the conda environment in terminal with conda installed use: <br/>
conda env create -f environment.yml

Then to activate the conda environment use:<br/>
conda activate llr_SP4
 
Once the enviroment is activated, install the analysis code with:<br/>
python setup.py install

-------------------------------------------------------------------------
To reproduce results of the paper, install this python package, download the data release and run the jupyter notebooks within the data release (reference).

--------------------------------------------------------------------------
Note: A LaTeX interpreter is required
If not installed please use 
Windows:
conda install -c conda-forge miktex 
Mac/ Linux:
conda install -c conda-forge texlive-core
