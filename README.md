# LFLE-UMamba: Local Feature Localization Extracation Vision Mamba-based UNet for Medical Image Segmentation <br> <span style="float: right"><sub><sup>BIBM 2024</sub></sup></span><br>

We propose *LFLE-UMamba* to address these issues, which
is a novel Mamba-based network. We introduce a Feature
Localization Block (FLB) to remove irrelevant background
information and locate effective regions in the foreground.
To better integrate the features extracted at different scales
by the network, we design the Feature Locate Visual Space
State Block (FLVSSB), which uses FLB to locate foreground
local information. LFLE-UMamba also uses a Local Feature
Extraction Module (LFFM) to extract and enhance foreground
regions’ local features. These two modules effectively improve
our network’s ability to capture local information and fuse
global semantic information.

![LFLE-UMamba](fig1.png)

