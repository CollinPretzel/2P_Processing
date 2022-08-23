# 2P_Processing
## Focused Ultrasound In-Vivo Glymphatic System Analysis
### Overview
These scripts are designed to take the data acquired from the BIO5 Two-Photon system, perform requisite pre-processing (e.g., Weiner filtering, Windowing Motion
Correction, and various registrations) and extract and identify all possible transverse arterioles and venules from bolus scans, as well as create reasonable
perivascular spaces around the transverse vessels. The classification of vessels and registration between different 3-dimensional scans currently have to be done
outside the bash processing script. Note: The bash processing script s currently designed for use on the University of Arizona HPC.

### General Pipeline
2P_DA.py <All_scan types>

2P_WF.py <All_scan_types>

2P_AR.py <3D_scan_types> <Potentially all...?>

2P_IR.py <All_scan_types>

2P_OT.py <3D_scan_types>

2P_VE.py <3D_scan_types>

2P_ID.py <3D_scan_types> <Bolus_scan(s)>

2P_PR.py <3D_scan_types>

2P_RB.py <3D_scan_types (moving)> <3D_scan_types (reference)> 

## Function Descriptions
### 2P_AR.py - Artifact Removal

Artifact removal for all Two-Photon images, which focuses on the Rhodamine scans as they demonstrate more significant features and then average the same FITC 
sections. The windowing correction method is adapted from: 

### 2P_DA.py - Data Acquisition

A method to compile a csv with relevant data from general screenshots.

### 2P_ID.py - Identification (arterioles vs. venules)

Designed to identify arterioles and venules using the bolus scans and extracted vessels. Assigns arterioles a value of 1, and venules a value of 2.

### 2P_IR.py - Internal Registration

Designed to align all images in a singular scan to reduce the impact of breathing or motion artifacts beyond the artifact removal function.

### 2P_OT.py - Otsu Thresholding

*Need to provide a description here*

### 2P_PR.py - Perivascular Regions

Designed to produce an approximate mask of the perivascular regions around vessels by analyze the centroid region of the vessels, creating a disk 8 or 10 um wider
around the vessel, then subtracting the vessel masks from these disks.

### 2P_proc.sh - Processing Bash Script

*Also need to provide another description here*

### 2P_RB.py - Registration Between (Scans)

*Here too!*

### 2P_VE.py - Vessel Extraction

Extracts the vessels from the thresholded scan using connected components analysis

### 2P_WF.py - Wiener Filtering

*Ahh one last one*
