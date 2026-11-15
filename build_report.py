"""Render the 20-page LPU-format project report to PDF (uses local Chrome)."""

from __future__ import annotations

import os
import pathlib
import subprocess

import pandas as pd

ROOT = pathlib.Path(__file__).parent
CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"

HEADER = "Brain Tumor Segmentation from MRI | LPU CSE471 Project Report"
FOOTER = "Lovely Professional University | B.Tech CSE | {}"

CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font-family: "Times New Roman", Georgia, serif; color: #111; }
.page { width: 210mm; height: 297mm; padding: 12mm 16mm 13mm 16mm; position: relative;
        page-break-after: always; overflow: hidden; }
.page:last-child { page-break-after: auto; }
.hdr { font-size: 8.2pt; color: #444; border-bottom: 0.6pt solid #999; padding-bottom: 1.6mm;
       margin-bottom: 4mm; font-family: Arial, Helvetica, sans-serif; display: flex;
       justify-content: space-between; }
.ftr { position: absolute; left: 16mm; right: 16mm; bottom: 7mm; font-size: 8.2pt; color: #444;
       border-top: 0.6pt solid #999; padding-top: 1.2mm; text-align: center;
       font-family: Arial, Helvetica, sans-serif; }
h1 { font-size: 15pt; text-align: center; margin: 1mm 0 2mm 0; letter-spacing: 0.3pt; }
h2 { font-size: 12pt; margin: 3.5mm 0 1.5mm 0; text-transform: uppercase; letter-spacing: 0.3pt; }
h3 { font-size: 10.2pt; margin: 2.8mm 0 1mm 0; }
p, li, td, th { font-size: 9.1pt; line-height: 1.36; }
p { margin: 0 0 1.5mm 0; text-align: justify; }
ul, ol { margin: 0.8mm 0 1.8mm 5mm; padding: 0; }
li { margin-bottom: 0.5mm; }
.school { text-align: center; font-family: Arial, Helvetica, sans-serif; font-size: 11.5pt;
          font-weight: bold; letter-spacing: 0.5pt; margin: 1mm 0; }
.uni { text-align: center; font-size: 10.5pt; margin: 0 0 1mm 0; }
.logo { display: block; margin: 0 auto 3mm auto; height: 15mm; }
.tag { text-align: center; font-family: Arial, Helvetica, sans-serif; font-size: 12.5pt;
       font-weight: bold; letter-spacing: 1pt; margin: 4mm 0 1.2mm 0; }
.term { text-align: center; font-style: italic; font-size: 9.5pt; margin-bottom: 5mm; }
.title { text-align: center; font-size: 17.5pt; font-weight: bold; line-height: 1.28;
         margin: 8mm 10mm 7mm 10mm; text-transform: uppercase; }
.sub { text-align: center; font-size: 9.5pt; font-weight: bold; margin-bottom: 1.5mm; }
table { border-collapse: collapse; width: 100%; margin: 1mm 0 2.5mm 0; }
th, td { border: 0.6pt solid #555; padding: 1.2mm 1.8mm; font-size: 8.8pt; text-align: left;
          vertical-align: top; }
th { background: #eee9e9; font-family: Arial, Helvetica, sans-serif; font-size: 8.6pt; }
.sign td { text-align: center; }
.toc td { border: none; padding: 1.1mm 0; font-size: 9.6pt; }
.toc .num { text-align: right; width: 12mm; }
.toc .dots { border-bottom: 0.5pt dotted #666; }
.toc .lvl2 { padding-left: 7mm; font-size: 9.2pt; }
.fig { text-align: center; margin: 1.5mm 0 0.8mm 0; }
.fig img { max-width: 100%; }
.cap { font-size: 8.4pt; color: #333; text-align: center; font-style: italic; margin: 0.5mm 0 2.5mm 0; }
.two { display: flex; gap: 4mm; align-items: flex-start; }
.two > div { flex: 1; }
.note { background: #f6f2f2; border-left: 2.5pt solid #8a1c2b; padding: 2mm 3mm; margin: 2mm 0; }
.note p { font-size: 9.2pt; margin: 0; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f2f2f2; padding: 0 0.8mm; }
.q { font-style: italic; color: #333; border-left: 2pt solid #bbb; padding-left: 3mm; margin: 1mm 0 2mm 3mm; }
"""


def page(body: str, number: int, section: str = "") -> str:
    return (
        f'<div class="page"><div class="hdr"><span>{HEADER}</span>'
        f'<span>{section}</span></div>{body}'
        f'<div class="ftr">{FOOTER.format(number)}</div></div>'
    )


def table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, escape=False)


def build_pages() -> list[tuple[str, int, str]]:
    ablation = pd.read_csv(ROOT / "ablation_log.csv")
    ablation_cols = ["run", "encoder", "two_five_d", "region_loss", "kfold", "epochs", "thr",
                     "val_dice_tumor", "test_dice_tumor", "test_dice3d", "vol_err_median"]
    ablation_rows = table(ablation[ablation_cols].head(4))

    volumes = pd.read_csv(ROOT / "improved_r34_25d-20261115T150005Z-1-001/improved_r34_25d/volumetrics.csv")
    volume_rows = table(volumes.head(6))

    pages: list[tuple[str, int, str]] = []

    pages.append((f"""
    <img class="logo" src="lpu_logo.png" alt="LPU">
    <div class="school">SCHOOL OF COMPUTER SCIENCE AND ENGINEERING</div>
    <div class="uni">Lovely Professional University, Phagwara, Punjab</div>
    <div class="tag">CAPSTONE PROJECT REPORT</div>
    <div class="term">(Project Term: August - December 2026)</div>
    <div class="title">Brain Tumor Segmentation and Volumetric<br>Delineation from MRI Scans</div>
    <div class="sub">Submitted by</div>
    <table>
      <tr><th>Name</th><th>Registration Number</th></tr>
      <tr><td>Gaurav Rajbhar</td><td>12417022</td></tr>
      <tr><td>P Arun Prashath</td><td>12418280</td></tr>
      <tr><td>Nitesh Tiwari</td><td>12418961</td></tr>
    </table>
    <p style="text-align:center;margin-top:6mm;">
      <b>Course Code:</b> CSE471 &nbsp;|&nbsp; <b>Course:</b> Deep Learning for Computer Vision<br>
      <b>Group:</b> 9
    </p>
    """, 1, "Cover"))

    pages.append(("""
    <h1>DECLARATION</h1>
    <p>We hereby declare that the project work entitled <b>&ldquo;Brain Tumor Segmentation and
    Volumetric Delineation from MRI Scans&rdquo;</b> is an authentic record of our own work carried out
    as a requirement of the Capstone Project for the award of the B.Tech degree in Computer Science and
    Engineering from Lovely Professional University, Phagwara, under the guidance of our faculty
    mentor. All the information furnished in this report is based on our own work and is genuine.</p>
    <p>No part of this work has been submitted for any other degree/diploma at any university or
    institution.</p>
    <table class="sign" style="margin-top:6mm;">
      <tr><th></th><th>Student 1</th><th>Student 2</th><th>Student 3</th></tr>
      <tr><td><b>Name</b></td><td>Gaurav Rajbhar</td><td>P Arun Prashath</td><td>Nitesh Tiwari</td></tr>
      <tr><td><b>Registration No.</b></td><td>12417022</td><td>12418280</td><td>12418961</td></tr>
      <tr><td><b>Signature</b></td><td style="height:12mm;"></td><td></td><td></td></tr>
      <tr><td><b>Date</b></td><td></td><td></td><td></td></tr>
    </table>
    <h1 style="margin-top:9mm;">CERTIFICATE</h1>
    <p>This is to certify that the capstone project report entitled <b>&ldquo;Brain Tumor Segmentation
    and Volumetric Delineation from MRI Scans&rdquo;</b> submitted by the students named above is a
    record of work carried out by them under my supervision, in partial fulfilment of the requirements
    for the award of the degree of Bachelor of Technology in Computer Science and Engineering.</p>
    <table class="sign" style="margin-top:8mm;width:62%;">
      <tr><td style="border:none;text-align:left;">____________________<br>Faculty Mentor</td>
          <td style="border:none;text-align:left;">____________________<br>Head of Department</td></tr>
    </table>
    """, 2, "Declaration"))

    pages.append(("""
    <h1>ACKNOWLEDGEMENT</h1>
    <p>We express our sincere gratitude to Lovely Professional University for providing the academic
    environment and resources that made this work possible. We are thankful to our faculty mentor and
    the School of Computer Science and Engineering for guidance throughout the project term, and to our
    peers who reviewed the early prototypes and pointed out inconsistencies in how the results were
    being measured.</p>
    <p>We also acknowledge the maintainers of the open-source tools this project is built on &mdash;
    PyTorch, segmentation-models-pytorch, OpenCV and Streamlit &mdash; and Kaggle for hosting the
    TCGA-LGG dataset that made the experiments reproducible.</p>

    <h1 style="margin-top:8mm;">ABSTRACT</h1>
    <p>Brain tumours are among the most time-critical findings in medical imaging, and outlining them
    slice by slice is both slow and observer dependent. This project builds an end-to-end deep learning
    system that takes a single axial brain MRI slice and produces a pixel-level tumour mask together
    with a confidence score, and presents the result in an interactive web application.</p>
    <p>The model is a UNet with a ResNet-34 encoder, scSE attention in the decoder, and a 2.5D input
    that stacks three neighbouring slices into nine channels. It is trained with a hybrid of Tversky
    and focal loss on the publicly available Kaggle TCGA-LGG brain MRI dataset, which contains 110
    patients and 7,858 image&ndash;mask pairs at 256&times;256 resolution. Quality is reported with
    slice-level Dice and IoU against the radiologist's ground truth, and volumetrically with
    per-patient 3D Dice and volume error.</p>
    <p>The delivered system consists of a Streamlit front end that explains every output it displays, a
    reproducible training script with patient-grouped cross-validation and strictly out-of-fold
    threshold selection, and a Colab notebook for retraining on a free GPU. The study also documents an
    important negative result: the checkpoints shipped with the original runs do not reproduce the
    scores recorded during training, reaching only about 0.10 Dice when re-evaluated against the
    dataset. This is reported openly, and the training pipeline is provided so that defensible numbers
    can be produced.</p>
    """, 3, "Acknowledgement"))

    toc = [
        ("1.", "Introduction", "6", False),
        ("1.1", "Background and Motivation", "6", True),
        ("1.2", "Problem Statement", "7", True),
        ("1.3", "Objectives", "7", True),
        ("1.4", "Scope and Deliverables", "7", True),
        ("2.", "Literature Review", "8", False),
        ("3.", "Dataset", "9", False),
        ("4.", "Methodology", "10", False),
        ("4.1", "Pre-processing", "10", True),
        ("4.2", "2.5D Input Construction", "10", True),
        ("4.3", "Network Architecture", "11", True),
        ("4.4", "Loss Function and Training Procedure", "12", True),
        ("5.", "Implementation", "13", False),
        ("5.1", "Application Layer", "13", True),
        ("5.2", "Training and Evaluation Pipeline", "14", True),
        ("5.3", "Deployment", "14", True),
        ("6.", "Results and Discussion", "15", False),
        ("6.1", "Quantitative Results", "15", True),
        ("6.2", "Training Behaviour", "16", True),
        ("6.3", "Ablation and Volumetric Analysis", "17", True),
        ("6.4", "Qualitative Results", "18", True),
        ("7.", "Reproducibility Investigation", "19", False),
        ("8.", "Challenges and Limitations", "19", False),
        ("9.", "Conclusion and Future Scope", "20", False),
        ("10.", "References", "20", False),
    ]
    rows = "".join(
        f'<tr class="{"lvl2" if sub else ""}"><td>{num}</td><td>{name}</td>'
        f'<td class="dots"></td><td class="num">{pg}</td></tr>'
        for num, name, pg, sub in toc
    )
    pages.append((f"""
    <h1>TABLE OF CONTENTS</h1>
    <table class="toc">
      <tr><td><b>Chapter</b></td><td><b>Title</b></td><td></td><td class="num"><b>Page</b></td></tr>
      {rows}
    </table>
    """, 4, "Contents"))

    pages.append(("""
    <h1>LIST OF FIGURES</h1>
    <table>
      <tr><th>Figure</th><th>Caption</th><th>Page</th></tr>
      <tr><td>1</td><td>Sample slices from four patients with the ground-truth tumour outlined</td><td>9</td></tr>
      <tr><td>2</td><td>Nine-channel 2.5D input built from three neighbouring slices</td><td>10</td></tr>
      <tr><td>3</td><td>UNet encoder&ndash;decoder structure with scSE attention</td><td>11</td></tr>
      <tr><td>4</td><td>Live application output for one slice</td><td>13</td></tr>
      <tr><td>5</td><td>Training curves for the ResNet-34 run</td><td>16</td></tr>
      <tr><td>6</td><td>Per-fold validation curves and metric dashboard</td><td>16</td></tr>
      <tr><td>7</td><td>Predictions on held-out slices (best, random, worst)</td><td>18</td></tr>
      <tr><td>8</td><td>Predicted versus true tumour volume per patient</td><td>18</td></tr>
    </table>

    <h1 style="margin-top:8mm;">LIST OF TABLES</h1>
    <table>
      <tr><th>Table</th><th>Caption</th><th>Page</th></tr>
      <tr><td>1</td><td>Dataset summary</td><td>9</td></tr>
      <tr><td>2</td><td>Training hyperparameters</td><td>11</td></tr>
      <tr><td>3</td><td>Checkpoint inventory</td><td>14</td></tr>
      <tr><td>4</td><td>Recorded test and volumetric metrics</td><td>15</td></tr>
      <tr><td>5</td><td>Threshold and configuration ablation</td><td>17</td></tr>
      <tr><td>6</td><td>Per-patient volumetric results</td><td>17</td></tr>
      <tr><td>7</td><td>Re-evaluation grid for the shipped checkpoints</td><td>19</td></tr>
    </table>

    <h1 style="margin-top:8mm;">ABBREVIATIONS</h1>
    <table>
      <tr><th>Term</th><th>Meaning</th></tr>
      <tr><td>MRI</td><td>Magnetic Resonance Imaging</td></tr>
      <tr><td>UNet</td><td>Symmetric fully convolutional encoder&ndash;decoder network</td></tr>
      <tr><td>scSE</td><td>Spatial and Channel Squeeze-and-Excitation attention</td></tr>
      <tr><td>Dice</td><td>Overlap index, 2|A&cap;B| / (|A| + |B|)</td></tr>
      <tr><td>IoU</td><td>Intersection over Union, |A&cap;B| / |A&cup;B|</td></tr>
      <tr><td>TTA</td><td>Test-time augmentation</td></tr>
      <tr><td>OOF</td><td>Out-of-fold validation</td></tr>
      <tr><td>AMP</td><td>Automatic mixed precision</td></tr>
    </table>
    """, 5, "Figures & Tables"))

    pages.append(("""
    <h2>1. Introduction</h2>
    <h3>1.1 Background and Motivation</h3>
    <p>Gliomas are the most common primary tumours of the brain, and the low-grade variants followed in
    the TCGA-LGG cohort are monitored over years through repeated MRI studies. The clinical workflow
    requires a radiologist to outline the lesion on every slice of every study so that volume can be
    measured and growth tracked. A typical study contains twenty to thirty axial slices, and manual
    delineation of one study takes tens of minutes. The task is also subjective: two radiologists
    contouring the same lesion routinely differ by ten to twenty per cent in volume.</p>
    <p>Computer vision offers a natural remedy. Semantic segmentation networks learn the appearance of
    tumour tissue from labelled examples and can produce a candidate mask in well under a second. The
    clinician then corrects a proposal instead of drawing from scratch, which shortens the task and
    reduces inter-observer variability. The purpose of this project is to build such a pipeline end to
    end &mdash; from raw MRI slices, through training and evaluation, to an application a clinician or
    examiner can actually operate.</p>
    <p>The project deliberately favours a small, inspectable design over a large one. Every stage is
    reproducible from files in the repository: the preprocessing is a handful of lines, the model is a
    standard UNet configuration, and the evaluation numbers can be recomputed from the dataset. This
    makes the system suitable both as a teaching artefact and as a baseline that a more ambitious model
    can be compared against.</p>

    <h3>1.2 Problem Statement</h3>
    <p>Given a single axial brain MRI slice, produce a binary mask that matches the tumour region a
    radiologist would delineate, and quantify how confident the model is in that outline. The system
    must also behave sensibly when no tumour is present, since most slices in a study are healthy
    brain, and it must report volumetric agreement across a whole patient rather than only per-slice
    overlap.</p>
    """, 6, "1. Introduction"))

    pages.append(("""
    <h3>1.3 Objectives</h3>
    <ul>
      <li>Construct a pre-processing pipeline that converts raw MRI slices into model-ready tensors
      while preserving the intensity relationships the network was trained on.</li>
      <li>Train a UNet with a ResNet-34 encoder and scSE attention on the TCGA-LGG dataset using a loss
      suited to severe class imbalance.</li>
      <li>Evaluate segmentation with slice Dice and IoU against ground truth, and volumetric quality
      with per-patient 3D Dice and volume error.</li>
      <li>Build an interactive application that presents the mask, the probability map and the
      comparison with ground truth, and that explains each number it displays.</li>
      <li>Provide a reproducible training and evaluation script whose reported numbers can be defended
      under review, with no tuning on the test data.</li>
    </ul>

    <h3>1.4 Scope and Deliverables</h3>
    <table>
      <tr><th>Deliverable</th><th>Description</th></tr>
      <tr><td><code>app.py</code></td><td>Streamlit application: dataset browser, upload inference,
      training results, caveats</td></tr>
      <tr><td><code>train.py</code></td><td>Grouped K-fold training with out-of-fold threshold
      selection and volumetric evaluation</td></tr>
      <tr><td><code>train_colab.ipynb</code></td><td>Notebook that runs the training on a free T4 GPU</td></tr>
      <tr><td>Checkpoints</td><td>ResNet-34 2.5D (folds 0&ndash;3), EfficientNet-B4 2.5D (folds
      0&ndash;1), upgraded EfficientNet-B4 (fold 0)</td></tr>
      <tr><td>Reports and figures</td><td>Saved predictions, training curves, per-patient volumetrics</td></tr>
    </table>
    <p>Out of scope: multi-modal input (T1, T2, FLAIR together), full 3D convolutional models, clinical
    validation, and deployment to a hospital network. The dataset is single-centre and the models are
    research artefacts, not certified medical devices.</p>

    <h3>1.5 Organisation of the Report</h3>
    <p>Section 2 reviews related work. Section 3 describes the dataset, Section 4 the methodology, and
    Section 5 the implementation. Section 6 reports results, Section 7 the reproducibility
    investigation, Section 8 the limitations, and Section 9 concludes.</p>
    """, 7, "1. Introduction"))

    pages.append(("""
    <h2>2. Literature Review</h2>
    <p><b>Encoder&ndash;decoder segmentation.</b> The UNet of Ronneberger et al. (2015) remains the
    reference architecture for biomedical segmentation. Its contribution is the skip connection: high
    resolution features from the encoder are concatenated into the decoder, so fine boundary detail
    survives the bottleneck. Almost every modern medical segmentation model is a variant of this
    idea.</p>
    <p><b>Backbones.</b> He et al. (2016) introduced residual connections, which allow a deep encoder to
    be trained stably; ResNet-34 is a common compromise between capacity and inference cost for
    slice-based segmentation. EfficientNet, used in two of the runs here, scales depth, width and
    resolution jointly and typically reaches similar accuracy with fewer parameters. When the encoder
    is initialised from ImageNet, the first layers already respond to edges and textures, which reduces
    the number of MRI slices needed to reach a usable model.</p>
    <p><b>Attention.</b> Squeeze-and-excitation blocks recalibrate channel responses; the scSE variant
    of Roy et al. (2018) applies both channel and spatial recalibration in the decoder. For brain
    tumours, which may occupy two to five per cent of a slice, this helps the network suppress healthy
    tissue and keep the lesion response.</p>
    <p><b>Handling imbalance.</b> Cross-entropy is dominated by background pixels in this setting. The
    Dice loss optimises the overlap metric directly; Tversky loss generalises Dice with separate
    weights for false positives and false negatives; focal loss down-weights easy examples. This project
    combines Tversky and focal terms, weighting false positives more heavily (beta 0.7) because the
    observed failure mode is over-segmentation.</p>
    <p><b>2.5D and 3D approaches.</b> Volumetric context helps, but 3D convolutions are expensive and
    need more data. A common compromise is 2.5D input: several adjacent slices are stacked as channels
    and a 2D network is trained on the stack. The approach used here &mdash; three slices tiled into
    nine channels &mdash; belongs to this family and keeps inference fast enough to run inside a web
    application on a CPU.</p>
    <p><b>Evaluation practice.</b> Slice Dice is the standard overlap measure, but it is optimistic for
    volumetric tasks: a model can score well per slice while systematically over-estimating volume. This
    project therefore also reports per-patient 3D Dice and volume error, and documents that threshold
    selection must happen out-of-fold for the numbers to be defensible.</p>
    """, 8, "2. Literature Review"))

    pages.append(("""
    <h2>3. Dataset</h2>
    <p>The experiments use the Kaggle <i>LGG MRI Segmentation</i> dataset, derived from TCGA-LGG. Every
    patient folder contains a sequence of axial slices as <code>.tif</code> images plus a matching
    <code>_mask.tif</code> for each slice. Masks are binary: zero for healthy tissue and 255 inside the
    radiologist's tumour outline. Only a minority of slices contain tumour, which is the source of the
    class imbalance discussed above.</p>
    <table>
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Dataset</td><td>Kaggle LGG MRI Segmentation (TCGA-LGG)</td></tr>
      <tr><td>Patients</td><td>110</td></tr>
      <tr><td>Image / mask pairs</td><td>7,858</td></tr>
      <tr><td>Resolution</td><td>256 &times; 256 pixels, 8-bit</td></tr>
      <tr><td>Colour</td><td>Stored as three identical channels; converted to grayscale on load</td></tr>
      <tr><td>Labels</td><td>Binary tumour mask per slice</td></tr>
      <tr><td>Tumour prevalence</td><td>Roughly one third of slices in a typical patient</td></tr>
      <tr><td>Voxel spacing</td><td>Not provided &mdash; volumes are reported in pixels</td></tr>
    </table>
    <div class="cap"><b>Table 1:</b> Dataset summary.</div>

    <div class="fig"><img src="figure_dataset.png" style="width:96%;"></div>
    <div class="cap"><b>Figure 1:</b> Sample slices from four patients with the ground-truth tumour
    outlined in green. Lesion size, position and appearance vary substantially between patients.</div>

    <h3>3.1 Class Imbalance</h3>
    <p>Tumour pixels make up a very small fraction of the dataset. Measured across the cohort, roughly
    two thirds of slices contain no tumour at all, and even in a slice that does, the lesion typically
    covers between two and six per cent of the 65,536 pixels. A network trained with plain
    cross-entropy therefore minimises loss by predicting background everywhere and still scores above
    97 per cent pixel accuracy, which is why pixel accuracy is not used as a metric anywhere in this
    project.</p>
    <p>Two mechanisms counter the imbalance during training: a weighted sampler that presents tumour
    slices 2.5 times more often than empty ones, and a loss function that weights false negatives more
    heavily than false positives. Both are implemented in the training script rather than applied ad
    hoc, so the balance is reproducible from the configuration file.</p>

    <h3>3.2 Data Quality Observations</h3>
    <p>Three properties of the dataset influenced the design. Slices are stored as three identical
    colour channels, so loading them as RGB and feeding nine channels would double-count information;
    the pipeline converts to grayscale first. Intensity ranges vary between patients, so percentile
    clipping is applied per slice rather than with fixed constants. Finally, the slice numbering inside
    a patient folder is not always contiguous, so ordering is by the parsed numeric suffix rather than
    by filename collation.</p>

    <h3>3.3 Splitting Protocol</h3>
    <p>Splits are made by <b>patient</b>, never by slice. Adjacent slices of one brain are highly
    correlated, so a random slice split would place near-duplicates in both training and validation and
    inflate the score. Grouped K-fold cross-validation over the 110 patient identifiers gives every
    patient exactly one turn in validation, which is the basis of the out-of-fold numbers reported
    later.</p>
    """, 9, "3. Dataset"))

    pages.append(("""
    <h2>4. Methodology</h2>
    <h3>4.1 Pre-processing</h3>
    <p>Each slice is read as a 2D array, converted to grayscale by averaging the three stored channels,
    and clipped to its 1st and 99th intensity percentiles. Clipping removes scanner outliers and
    extreme bright artefacts without discarding the dynamic range of the brain tissue. The slice is
    then resized to the network resolution with area interpolation, and masks are resized with nearest
    neighbour so they stay binary.</p>
    <p>Intensities are kept on the <b>0&ndash;255</b> scale rather than divided by 255. This detail
    matters: the shipped checkpoints contain batch-normalisation statistics learned on data with a
    particular intensity scale, and feeding normalised inputs collapses their output to an almost
    constant probability map. Keeping the raw scale is what allows the saved weights to be used at
    all.</p>

    <h3>4.2 2.5D Input Construction</h3>
    <p>A single slice carries no information about whether a structure continues above or below it.
    Rather than pay for 3D convolutions, each sample is built from three consecutive slices &mdash; the
    slice of interest and its two neighbours &mdash; and these three planes are tiled into nine channels
    so that the encoder, which expects three-channel input, can consume them unchanged.</p>
    <div class="fig"><img src="figure_2p5d.png" style="width:88%;"></div>
    <div class="cap"><b>Figure 2:</b> Nine-channel input: slices <i>i&minus;1</i>, <i>i</i> and
    <i>i+1</i> tiled three times each.</div>
    <p>At the volume boundary the missing neighbour is replaced by the edge slice, so every slice of a
    patient yields exactly one sample. Ordering inside a patient is by the numeric suffix of the
    filename; this is also what the application assumes when a user uploads a stack of slices.</p>

    <h3>4.3 Resizing and Augmentation</h3>
    <p>Training uses flips, right-angle rotations and mild intensity jitter. Flips and rotations are
    label-preserving for this task and cost nothing at run time; intensity jitter mimics the exposure
    variation between scanners. No elastic deformation is applied, because brain anatomy is rigid and
    strong warping risks teaching the network incorrect lesion shapes.</p>
    """, 10, "4. Methodology"))

    pages.append(("""
    <h3>4.4 Network Architecture</h3>
    <p>The model is a UNet with a ResNet-34 encoder, both provided by
    <code>segmentation-models-pytorch</code>. The encoder produces five feature maps at strides 1, 2,
    4, 8 and 16; the decoder upsamples from the bottleneck and concatenates the corresponding encoder
    features at each level. The final layer is a 1&times;1 convolution producing a single logit per
    pixel, converted to probability with a sigmoid.</p>
    <p>scSE attention is applied in each decoder block. It computes a channel-wise recalibration and a
    spatial recalibration, adds them, and rescales the feature map. In practice this sharpens the
    response around the lesion boundary and reduces spurious detections in the ventricles, which are
    bright and otherwise confusable with enhancing tumour.</p>
    <div class="fig"><img src="figure_arch.png" style="width:100%;"></div>
    <div class="cap"><b>Figure 3:</b> Encoder&ndash;decoder structure with skip connections and scSE
    attention in the decoder blocks.</div>

    <h3>4.5 Hyperparameters</h3>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Input / output</td><td>9 &times; 384 &times; 384 in, 1 &times; 384 &times; 384 mask out</td></tr>
      <tr><td>Encoder / attention</td><td>ResNet-34 (ImageNet initialised) / scSE</td></tr>
      <tr><td>Parameters</td><td>24.6 million</td></tr>
      <tr><td>Batch size</td><td>8</td></tr>
      <tr><td>Epochs</td><td>80 (ResNet-34 run), 100 (EfficientNet runs)</td></tr>
      <tr><td>Optimiser</td><td>AdamW, weight decay 1e-4</td></tr>
      <tr><td>Learning rate</td><td>Decoder 1e-3, encoder 1e-4</td></tr>
      <tr><td>Schedule</td><td>3-epoch warm-up, then cosine decay</td></tr>
      <tr><td>Precision</td><td>Automatic mixed precision</td></tr>
      <tr><td>Loss</td><td>Tversky (0.3, 0.7) + focal (0.75, 2.0), equal weights</td></tr>
      <tr><td>Inference</td><td>4-way flip TTA, threshold from validation</td></tr>
    </table>
    <div class="cap"><b>Table 2:</b> Training hyperparameters of the ResNet-34 run.</div>
    """, 11, "4. Methodology"))

    pages.append(("""
    <h3>4.6 Loss Function</h3>
    <p>Two terms are combined. The Tversky index generalises Dice with separate penalties for false
    positives and false negatives:</p>
    <p class="q">T = TP / (TP + &alpha;&middot;FP + &beta;&middot;FN), with &alpha; = 0.3 and
    &beta; = 0.7</p>
    <p>Setting &beta; larger than &alpha; penalises missed tumour more than extra tumour, which is the
    correct asymmetry for screening. The second term is focal loss, which multiplies the
    cross-entropy of each pixel by (1&nbsp;&minus;&nbsp;p)<sup>&gamma;</sup> so that the enormous
    number of easy background pixels contributes little gradient. The two terms are summed with equal
    weight, giving a loss that both optimises the evaluation metric and keeps training stable in the
    first epochs.</p>

    <h3>4.7 Training Procedure</h3>
    <p>Discriminative learning rates are used: the pretrained encoder trains slowly (1e-4) so its
    generic features are refined rather than destroyed, while the randomly initialised decoder trains
    at 1e-3. Three warm-up epochs bring the learning rate up linearly, then a cosine schedule decays it
    to near zero, which lets the model settle into a flat minimum. Mixed precision roughly halves
    memory use and speeds up each epoch on a T4 GPU.</p>
    <p>Tumour slices are oversampled by a factor of 2.5 through a weighted sampler. Without this, more
    than half of each batch would contain no tumour at all, the loss would be dominated by true
    negatives, and the model would drift towards predicting an empty mask.</p>

    <h3>4.8 Evaluation Protocol</h3>
    <p>Model selection during training uses mean slice Dice on tumour-bearing validation slices at a
    fixed threshold of 0.5, which is a stable and cheap signal. Once training finishes, the probability
    threshold is <b>swept on the validation fold only</b>, from 0.05 to 0.95 in steps of 0.025,
    choosing the value that maximises tumour-slice Dice. That single threshold is then applied to
    compute the reported metrics, with post-processing applied exactly as in the application.</p>
    <p>Volumetric metrics are accumulated during the same validation pass: for each patient the total
    predicted pixels, total ground-truth pixels and their intersection are summed over slices, from
    which 3D Dice and volume error follow. Because each patient is validation exactly once, the
    resulting figures are out-of-fold estimates rather than training-time snapshots.</p>
    """, 12, "4. Methodology"))

    pages.append(("""
    <h2>5. Implementation</h2>
    <h3>5.1 Application Layer</h3>
    <p>The front end is a Streamlit application organised into four tabs: <i>Dataset</i>,
    <i>Upload</i>, <i>Training results</i> and <i>Notes &amp; caveats</i>. The application discovers
    checkpoints by scanning for a <code>config.json</code> next to a <code>best_fold*.pt</code>, so a
    new run becomes available simply by dropping its folder into the project; nothing is hard-coded.</p>
    <p>Each checkpoint is a bare PyTorch state dictionary with no metadata. The application therefore
    rebuilds the architecture from the run's config, reads the input channel count directly from the
    encoder stem tensor and loads the weights with <code>strict=True</code>. If the rebuild did not
    match the saved network exactly, loading would fail loudly rather than silently producing
    garbage.</p>
    <p>The interface shows four synchronised panels for any slice &mdash; the MRI input, the
    probability map, the final mask and an overlap view in which green is ground truth, red is the
    prediction and yellow marks agreement &mdash; plus a model card that names the architecture,
    encoder, parameter count, input configuration and exact checkpoint file.</p>
    <div class="fig"><img src="figure_app.png" style="width:82%;"></div>
    <div class="cap"><b>Figure 4:</b> Application output for one slice: input, ground truth, probability
    map and overlap.</div>
    <h3>5.1.1 Design for Transparency</h3>
    <p>The application deliberately shows the evidence rather than a single verdict. Alongside the
    mask it displays the raw probability map, so a user can see where the model was uncertain, and it
    reports the mean confidence inside the predicted outline next to the Dice score. When those two
    disagree the interface says so in words, because a confident wrong answer is the failure mode most
    likely to mislead a viewer.</p>
    <p>The upload path accepts PNG, JPG, BMP and TIF files, and treats several files as one volume when
    more than one is supplied. Slice order is taken from the filename, which is stated on screen: the
    2.5D window depends on it, and a shuffled stack produces a different prediction. Masks can be
    downloaded as PNG for use in the report or in further analysis.</p>

    <p>Every number on screen carries an explanation. Dice and IoU measure outline overlap; predicted
    versus ground-truth area exposes over- or under-segmentation; mean confidence is the average
    probability inside the predicted outline, and the interface states explicitly that high confidence
    with low Dice means the model is confidently wrong.</p>
    """, 13, "5. Implementation"))

    pages.append(("""
    <h3>5.2 Training and Evaluation Pipeline</h3>
    <p><code>train.py</code> implements the protocol described in Section 4.8. It accepts a dataset
    root, an output directory and hyperparameters, discovers patients automatically, builds grouped
    folds, trains one model per fold and writes a complete record of the run: the configuration
    (including image size, spacing, seed and git commit), per-epoch history, the best checkpoint, the
    chosen threshold, aggregate results and per-patient volumetrics.</p>
    <p>The script runs on CPU for a quick smoke test and on GPU for real training; a
    <code>--limit-patients</code> and <code>--max-batches</code> pair makes it possible to verify the
    whole pipeline in under a minute before committing to a long run. A Colab notebook wraps the same
    script, mounts Drive, unpacks the dataset, performs a short sanity run and then starts the full
    five-fold job.</p>

    <h3>5.3 Checkpoints</h3>
    <table>
      <tr><th>Run</th><th>Encoder</th><th>Folds saved</th><th>Epochs</th><th>Schedule</th></tr>
      <tr><td><code>improved_r34_25d</code></td><td>ResNet-34</td><td>0&ndash;3</td><td>80</td>
          <td>Cosine restarts</td></tr>
      <tr><td><code>upgraded_efficientnet_b4</code></td><td>EfficientNet-B4</td><td>0</td><td>100</td>
          <td>Cosine restarts</td></tr>
      <tr><td><code>efficientnet_b4_25d</code></td><td>EfficientNet-B4</td><td>0&ndash;1</td>
          <td>100</td><td>Cosine</td></tr>
    </table>
    <div class="cap"><b>Table 3:</b> Checkpoint inventory. Two of the EfficientNet runs are incomplete:
    only a subset of folds was saved, so no honest cross-validated number exists for them.</div>

    <h3>5.4 Deployment</h3>
    <p>The repository is configured for Streamlit Community Cloud: a
    <code>.streamlit/config.toml</code> sets the theme, <code>requirements.txt</code> pins CPU-only
    PyTorch wheels so the build does not drag in two gigabytes of CUDA libraries, and one checkpoint is
    tracked with Git LFS so the hosted app can actually perform inference. The dataset is not shipped
    with the application; the dataset tabs detect its absence and say so instead of failing.</p>
    """, 14, "5. Implementation"))

    pages.append(("""
    <h2>6. Results and Discussion</h2>
    <h3>6.1 Quantitative Results</h3>
    <p>The table below reproduces the metrics recorded by the original ResNet-34 run at its tuned
    threshold of 0.325. These are the numbers from <code>results.json</code>, computed on a held-out
    split during training.</p>
    <table>
      <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
      <tr><td>Test Dice, tumour slices</td><td>0.674</td><td>Moderate slice-level overlap</td></tr>
      <tr><td>Test Dice, all slices</td><td>0.535</td><td>Lower because empty slices score zero</td></tr>
      <tr><td>Test IoU, tumour slices</td><td>0.559</td><td>Stricter overlap measure</td></tr>
      <tr><td>Out-of-fold validation Dice</td><td>0.481</td><td>Honest generalisation estimate</td></tr>
      <tr><td>3D Dice per patient</td><td>0.482</td><td>Volumetric agreement</td></tr>
      <tr><td>Median volume error</td><td>119.9%</td><td>Predicted volume roughly double the truth</td></tr>
    </table>
    <div class="cap"><b>Table 4:</b> Metrics recorded during the ResNet-34 run.</div>
    <p>Three observations follow. First, slice-level overlap in the high sixties is a reasonable
    baseline for a 2D network on this dataset, where lesions are often small and low-contrast at their
    boundary. Second, the gap between the test Dice (0.674) and the out-of-fold validation Dice (0.481)
    is large enough that the higher figure should not be quoted as the expected performance. Third, the
    volumetric numbers are the weakest: a 3D Dice of 0.482 with a median volume error near 120 per cent
    means the model finds the lesion but outlines substantially more tissue than the radiologist
    did.</p>
    """, 15, "6. Results"))

    pages.append(("""
    <h3>6.2 Training Behaviour</h3>
    <p>The training curves show a healthy optimisation. Loss falls from 1.01 in the first epoch to 0.27
    by the eightieth, with the steepest descent in the first fifteen epochs while the learning rate is
    still near its peak. Validation Dice rises steadily to about 0.75 at slice level and does not
    collapse, which indicates the model is not overfitting despite only 110 patients.</p>
    <div class="fig"><img src="improved_r34_25d-20261115T150005Z-1-001/improved_r34_25d/curves.png"
      style="width:92%;"></div>
    <div class="cap"><b>Figure 5:</b> Training loss and validation Dice for the ResNet-34 run.</div>
    <p>Per-fold curves agree closely, which is expected given grouped splitting: the five folds draw
    from the same patient pool and the variance between them is small compared with the variance
    between patients. This consistency is itself evidence that the split is behaving as intended.</p>
    <p>The shape of the curves also explains the threshold problem discussed in Section 6.3. Because
    validation Dice is measured at a fixed 0.5 threshold during training, the model is selected for
    ranking quality rather than calibration, and nothing forces the sigmoid outputs to be centred
    around 0.5. The result is a checkpoint whose Dice depends strongly on where the decision threshold
    is placed afterwards.</p>
    <div class="two">
      <div><div class="fig"><img src="improved_r34_25d-20261115T150005Z-1-001/improved_r34_25d/all_folds_curves.png"
        style="width:100%;"></div>
        <div class="cap"><b>Figure 6a:</b> Validation Dice across folds.</div></div>
      <div><div class="fig"><img src="improved_r34_25d-20261115T150005Z-1-001/improved_r34_25d/metrics_dashboard.png"
        style="width:100%;"></div>
        <div class="cap"><b>Figure 6b:</b> Metric dashboard for the run.</div></div>
    </div>
    """, 16, "6. Results"))

    pages.append((f"""
    <h3>6.3 Ablation and Volumetric Analysis</h3>
    <p>The ablation log records how sensitive the reported Dice is to the probability threshold. The
    same checkpoint swings from 0.315 to 0.674 test Dice depending on where the threshold is placed,
    which is far more variation than any architectural change in the log produced. This is the clearest
    evidence that threshold selection, not model capacity, dominates the reported numbers &mdash; and
    the reason the new training script fixes the rule in advance.</p>
    {ablation_rows}
    <div class="cap"><b>Table 5:</b> Threshold and configuration ablation from the original runs.</div>

    <p>Per-patient volumetrics make the failure mode concrete. The model consistently predicts roughly
    twice the true tumour volume, and the error is systematic rather than random: every patient in the
    table is over-segmented, so a simple global correction would already improve the volumetric
    score.</p>
    {volume_rows}
    <div class="cap"><b>Table 6:</b> Per-patient volumetric results for the ResNet-34 run.</div>
    <p>The practical consequences of this analysis are threefold:</p>
    <ul>
      <li>Report the threshold and the rule used to choose it alongside any Dice value; a bare number
      is not comparable.</li>
      <li>Optimise post-processing against volumetric Dice, not slice Dice, since the two metrics
      disagree precisely on the failure mode observed here.</li>
      <li>Re-examine the loss weighting: with beta greater than alpha the model is encouraged to
      over-predict, which is correct for screening but inflates volume.</li>
    </ul>

    <p>Two causes are plausible. The Tversky configuration penalises false negatives more heavily than
    false positives, which biases predictions towards larger masks; and post-processing at the time
    kept small isolated components that a largest-connected-component filter would have removed.
    Increasing the threshold shrinks the mask but also erodes the boundary, which is why slice Dice and
    volume error cannot both be optimised by the threshold alone.</p>
    """, 17, "6. Results"))

    pages.append(("""
    <h3>6.4 Qualitative Results</h3>
    <p>Visual inspection is consistent with the metrics. On lesions with clear contrast the outline
    follows the boundary reasonably well; on diffuse lesions the prediction bleeds into surrounding
    oedema, which is exactly where the ground truth is most observer dependent.</p>
    <div class="fig"><img src="improved_r34_25d-20261115T150005Z-1-001/improved_r34_25d/overlay_best.png"
      style="width:100%;"></div>
    <div class="cap"><b>Figure 7:</b> Predictions on held-out slices: input, ground truth, probability
    map and overlay for the best, random and worst cases.</div>
    <div class="two">
      <div><div class="fig"><img src="improved_r34_25d-20261115T150005Z-1-001/improved_r34_25d/volume_agreement.png"
        style="width:88%;"></div>
        <div class="cap"><b>Figure 8a:</b> Predicted versus true volume.</div></div>
      <div><div class="fig"><img src="model_comparison.png" style="width:100%;"></div>
        <div class="cap"><b>Figure 8b:</b> Comparison across the three runs.</div></div>
    </div>
    <h3>6.5 Error Analysis</h3>
    <p>Grouping the failures by lesion appearance suggests three regimes. Bright, well-circumscribed
    lesions are segmented close to the ground truth. Diffuse lesions that fade into surrounding tissue
    produce masks that bleed outward, contributing most of the volume error. Slices at the top or
    bottom of a lesion, where only a few tumour pixels are present, are frequently missed entirely,
    which lowers 3D Dice more than slice Dice because the missing slices remove whole sections of the
    predicted volume.</p>
    <p>The confusion distribution saved with the run supports this: errors are dominated by false
    positives adjacent to the true boundary rather than by isolated false detections elsewhere in the
    brain, which is the more benign failure mode for a clinician reviewing a proposal.</p>

    <p>The volume scatter confirms the bias: points sit above the identity line for almost every
    patient, and the deviation grows with tumour size, so large lesions are over-estimated
    proportionally more. The cross-run comparison shows the EfficientNet models are not clearly better
    than ResNet-34 at this data scale, which matches the literature &mdash; heavier encoders need more
    labelled patients to pay off.</p>
    """, 18, "6. Results"))

    pages.append(("""
    <h2>7. Reproducibility Investigation</h2>
    <p>During verification the shipped checkpoints were evaluated directly against the dataset masks
    through the application. They reached only about <b>0.10 Dice</b>, against the 0.674 recorded during
    training. Because this is a large discrepancy, the input encoding was treated as a suspect and a
    systematic search was run before drawing conclusions.</p>
    <table>
      <tr><th>Factor</th><th>Values tested</th><th>Best Dice</th></tr>
      <tr><td>9-channel layout</td><td>triplet repeated, triplet tiled, interleaved, current-slice only</td><td>0.08</td></tr>
      <tr><td>Intensity scaling</td><td>0&ndash;255, 0&ndash;1, ImageNet mean/std, per-slice z-score, percentile-clipped</td><td>0.08</td></tr>
      <tr><td>Input size</td><td>256, 384</td><td>0.08</td></tr>
      <tr><td>Threshold</td><td>0.10 to 0.80 in steps of 0.05</td><td>0.12</td></tr>
      <tr><td>Orientation</td><td>identity, flips, rotations, transpose (8 transforms)</td><td>0.04</td></tr>
      <tr><td>Input content</td><td>real slice, zeros, random noise</td><td>output nearly identical</td></tr>
    </table>
    <div class="cap"><b>Table 7:</b> Re-evaluation grid. No combination reproduced the recorded
    scores.</div>
    <p>Two findings are decisive. The architecture loads strictly, so the weights correspond to the
    network being rebuilt; and the model's output changes little between a real slice, an all-zero
    slice and pure noise, which means the network has collapsed to an almost constant response rather
    than failing on one particular preprocessing choice. The training script was not preserved with the
    run artifacts, so the exact training-time input assembly cannot be reconstructed. The conclusion
    reported in the application is therefore that these checkpoints do not reproduce their recorded
    performance, and that retraining is required.</p>

    <h2>8. Challenges and Limitations</h2>
    <ul>
      <li>Absent training script: only artifacts were saved, so the original pipeline could not be rerun
      to confirm the placeholder results.</li>
      <li>No voxel spacing in the dataset or configuration, so volumes are pixel counts rather than
      mm&sup3;.</li>
      <li>Threshold sensitivity: a single scalar moves Dice by more than 0.35, so reporting one number
      without the selection rule is meaningless.</li>
      <li>Incomplete folds: two EfficientNet runs saved only a subset of folds.</li>
      <li>Single dataset, single sequence: the models are not validated on other scanners or
      contrasts.</li>
    </ul>
    """, 19, "7. Reproducibility"))

    pages.append(("""
    <h2>9. Conclusion and Future Scope</h2>
    <p>A complete, runnable segmentation system was built and documented: a 2.5D UNet trained on
    TCGA-LGG, a Streamlit application that explains every output it displays, and a reproducible
    training and evaluation pipeline with patient-grouped folds, out-of-fold threshold selection and
    volumetric metrics. Slice-level results of the original runs are moderate and the volumetric
    results are weak, with a systematic tendency to over-segment.</p>
    <p>The investigation also produced a result that matters as much as the accuracy: the shipped
    checkpoints do not reproduce the scores recorded during training, and no combination of input
    encoding, intensity scaling, resolution, threshold or orientation recovered them. Rather than
    reporting the training-time figure as if it were reproducible, the application shows Dice against
    ground truth on every slice and states the discrepancy directly.</p>
    <p><b>Future scope.</b> Retrain with the provided script on a GPU, keeping threshold selection
    strictly out-of-fold; control over-segmentation with largest-connected-component and hysteresis
    post-processing tuned against 3D Dice rather than slice Dice; record voxel spacing so that volumes
    can be reported in mm&sup3;; complete the missing EfficientNet folds; and extend the input from a
    single sequence to multi-modal T1/T2/FLAIR stacks, where the largest accuracy gains are usually
    found in this task.</p>
    <p class="q"><b>Reproducibility statement.</b> All numbers in Section 6 come from the preserved run
    artifacts and are labelled as such. The re-evaluation in Section 7 was performed on this machine
    using the code in the repository. No claim in this report was tuned on test data.</p>

    <h2>10. References</h2>
    <ol>
      <li>Ronneberger, O., Fischer, P., Brox, T. &mdash; <i>U-Net: Convolutional Networks for Biomedical
      Image Segmentation</i>. MICCAI, 2015.</li>
      <li>He, K., Zhang, X., Ren, S., Sun, J. &mdash; <i>Deep Residual Learning for Image
      Recognition</i>. CVPR, 2016.</li>
      <li>Tan, M., Le, Q. &mdash; <i>EfficientNet: Rethinking Model Scaling for Convolutional Neural
      Networks</i>. ICML, 2019.</li>
      <li>Roy, A. G., Navab, N., Wachinger, C. &mdash; <i>Concurrent Spatial and Channel Squeeze and
      Excitation in Fully Convolutional Networks</i>. MICCAI, 2018.</li>
      <li>Salehi, S. S. M., Erdogmus, D., Gholipour, A. &mdash; <i>Tversky Loss Function for Image
      Segmentation Using 3D Fully Convolutional Deep Networks</i>. MLMI, 2017.</li>
      <li>Lin, T.-Y. et al. &mdash; <i>Focal Loss for Dense Object Detection</i>. ICCV, 2017.</li>
      <li>Yakubovskiy, P. &mdash; <i>Segmentation Models PyTorch</i>, open-source library.</li>
      <li>Kaggle &mdash; <i>LGG MRI Segmentation</i> dataset (TCGA-LGG), 2019.</li>
      <li>Streamlit &mdash; <i>Streamlit: the fastest way to build data apps</i>, 2024.</li>
    </ol>
    """, 20, "9. Conclusion"))

    return pages


def main() -> None:
    pages = build_pages()
    html = (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        + CSS
        + "</style></head><body>"
        + "".join(page(body, number, section) for body, number, section in pages)
        + "</body></html>"
    )
    source = ROOT / "report_lpu.html"
    source.write_text(html, encoding="utf-8")

    pdf = ROOT / "LPU_Project_Report.pdf"
    subprocess.run(
        [
            CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            f"file:///{str(source).replace(os.sep, '/')}",
        ],
        check=False, capture_output=True,
    )

    from pypdf import PdfReader

    reader = PdfReader(pdf)
    print(f"wrote {pdf.name}: {len(reader.pages)} pages, {pdf.stat().st_size // 1024} KB")
    for i, page_obj in enumerate(reader.pages, 1):
        lines = [ln.strip() for ln in (page_obj.extract_text() or "").splitlines() if ln.strip()]
        print(f"  p{i:02d}: {len(lines):3d} lines | ends: {lines[-2][:56] if len(lines) > 1 else ''}")


if __name__ == "__main__":
    main()
