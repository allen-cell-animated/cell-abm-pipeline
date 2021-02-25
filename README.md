# cell-ABM

Useful scripts for AICS cell-scale agent-based models project

---

# Visualize PhysiCell outputs in Simularium

Create an environment, install `simulariumio` (https://github.com/allen-cell-animated/simulariumio), and run the conversion script:
```python
conda create -n physicell_simularium python=3
conda activate physicell_simularium
pip install 'simulariumio[physicell]'
python [path_to_this_repo]/visualization/physicell_to_simularium.py [path_to_PhysiCell_project]/output physicell0
```
this will create a file 'physicell0.simularium' in your working directory, drag and drop this at https://simularium.allencell.org/viewer

