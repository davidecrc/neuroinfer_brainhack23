# *NeuroInfer*

NeuroInfer (NI) is a tool for drawing Bayesian inferences on the associations between cognitive processes and brain activations based on meta-analytic data from fMRI published studies. In a nutshell, it allows neuroscientists to interpret their results and assess the reliability of their hypotheses. 
Compared to available tools (e.g., NeuroSynth, NeuroQuery, BACON), NI implements a more sophisticated Bayesian analysis (based on Bayesian confirmation theory). 
The NI framework includes a database of published fMRI studies and related metadata (updated in 2018 and shared with NeuroSynth and NeuroQuery), a vocabulary of terms, and a Python-based algorithm to perform the analysis. 

## GUI: A web-visualizer

![neuroinfer_gui](./neuroinfer/images/neuroinfer_gui.png "Title")


## Example of Uses:
First, be sure all the needed dependencies are correctly installed:

     $ pip install -r neuroinfer/requirements.txt


In order to open the gui on your own browser, just run the server.py script from your terminal:

     $ cd /your/path/to/neuroinfer/root
     $ pyhon -m neuroinfer.server

The python script analysis can be separatedely launched as follows:

     $ pyhon MainScript.py
     Enter cog(s) here (if more than one separate them with a comma): memory
     Enter a prior between 0.001 and 1.000 associated with memory: 0.1
     The analysis will be run on: memory with prior probability: 0.1.
     Do you want to insert coordinates (c) or the area (a)? a
     Choose an area from the list: 150
     Enter ROI radius here (in mm): 4
     Selected area: 150Radius: 4.0 mm.
     Specify the Bayesian confirmation measure you want to use in the analysis (type one of the following letter):
     'a' for Bayes' Factor;
     'b' for difference measure;
     'c' for ratio measure.
     a
     
Then the result is saved:

     $ Time in min: 0.03
     $ Results saved to: /home/saram/PhD/neuroinfer_brainhack23/neuroinfer/results/results_BHL_area150.pickle

## To DO:

**Analysis:**
  - [ ] general optimization of the analysis code
  - [x] implementation searchlight analysis
  - [x] implementation whole-brain analysis
  - [x] implementation of atlas-guided analysis

**Results:**
   - _**back-end:**_
       - [ ] MainScript.py should be turned into __init__.py
       - [ ] UserInputs.py should be turned into a listener of the script form
       - [x] index.html should be automatically open when the python script is run
       - [ ] create a python script to listen to the scripts output and lunch the anlaysis.

   - _**front-end:**_
        - [ ] add a loading image in the web page while the analysis is run (and possibly estimate the time)
        - [ ] add an info box in the web page if the python script is detached
        - [x] fix mni template showing when loading the page
        - [ ] implementing labeling overimposed to brain template (HTML5 & canvas)
        - [ ] implementing multi plots/results viewing/selection

## Folder structure

    ./neuroinfer/
    ├── code/
    │   │── plot_generator.py       # a template using flask_cors to interact with the javascript
    │   │── DataLoading.py
    │   │── BayesianAnalysis.py     # core bayesian anlaysis work
    │   │── SummaryReport.py
    │   │── UserInputs.py           # expects input from the command-line
    │   └── MainScript.py           # main script (to be moved inside __init__.py)
    ├── html/
    │   │── index.html              # this should be automatically open when the python 
    │   │── main.js                 # provide the scripts to itneract with python
    │   │── styles.css
    │   └── viewer-config.js        # initialize neuroimaging div with mni template
    ├── templates/
    │   └── {...}.nii.gz
    ├── data/                       # it contains the bibliography data needed to compute the analysis
    │   │── featuures7.npz
    │   │── metadata7.csv
    │   │── vocabulary7.txt
    │   │── data-neurosynth_version-7_metadata.tsv
    │   └── data-neurosynth_version-7_coordinates.tsv
    ├── rii-mango-Papaya-#####/     # from papaya repo. it hosts the papaya visualizer js
    ├── images/     
    └── README.md
