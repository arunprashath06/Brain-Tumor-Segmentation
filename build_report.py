"""Render the 4-page LPU-format project report from HTML (uses local Chrome)."""

from __future__ import annotations

import os
import pathlib
import subprocess

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
       margin-bottom: 4mm; font-family: Arial, Helvetica, sans-serif; }
.ftr { position: absolute; left: 16mm; right: 16mm; bottom: 7mm; font-size: 8.2pt; color: #444;
       border-top: 0.6pt solid #999; padding-top: 1.2mm; text-align: center;
       font-family: Arial, Helvetica, sans-serif; }
h1 { font-size: 15pt; text-align: center; margin: 1mm 0 2mm 0; letter-spacing: 0.3pt; }
h2 { font-size: 11pt; margin: 3mm 0 1.2mm 0; text-transform: uppercase; letter-spacing: 0.3pt; }
p, li, td { font-size: 9.2pt; line-height: 1.38; }
p { margin: 0 0 1.6mm 0; text-align: justify; }
ul, ol { margin: 0.8mm 0 1.6mm 4.5mm; padding: 0; }
li { margin-bottom: 0.4mm; }
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
table { border-collapse: collapse; width: 100%; margin: 0.8mm 0 2mm 0; }
th, td { border: 0.6pt solid #555; padding: 1.1mm 1.6mm; font-size: 8.6pt; text-align: left; }
th { background: #eee9e9; font-family: Arial, Helvetica, sans-serif; font-size: 8.4pt; }
.sign td { text-align: center; }
.toc td { border: none; padding: 0.9mm 0; font-size: 9.2pt; }
.toc .num { text-align: right; width: 12mm; }
.toc .dots { border-bottom: 0.5pt dotted #666; }
.fig { text-align: center; margin: 1mm 0 0.5mm 0; }
.fig img { max-width: 100%; }
.cap { font-size: 8.2pt; color: #333; text-align: center; font-style: italic; margin: 0.4mm 0 1.8mm 0; }
.two { display: flex; gap: 3.5mm; align-items: flex-start; }
.two > div { flex: 1; }
.note { background: #f6f2f2; border-left: 2.5pt solid #8a1c2b; padding: 1.6mm 2.5mm; margin: 1.5mm 0; }
.note p { font-size: 8.8pt; margin: 0; }
code { font-family: Consolas, monospace; font-size: 8.4pt; background: #f2f2f2; padding: 0 0.8mm; }
"""


def page(body: str, number: int) -> str:
    return (
        f'<div class="page"><div class="hdr">{HEADER}</div>{body}'
        f'<div class="ftr">{FOOTER.format(number)}</div></div>'
    )


def build_html() -> str:
    cover = """
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
    <p style="text-align:center;margin-top:5mm;">
      <b>Course Code:</b> CSE471 &nbsp;|&nbsp; <b>Course:</b> Deep Learning for Computer Vision<br>
      <b>Group:</b> 9
    </p>
    """

    declaration = """
    <h1>DECLARATION</h1>
    <p>We hereby declare that the project work entitled <b>&ldquo;Brain Tumor Segmentation and
    Volumetric Delineation from MRI Scans&rdquo;</b> is an authentic record of our own work carried
    out as a requirement of the Capstone Project for the award of the B.Tech degree in Computer
    Science and Engineering from Lovely Professional University, Phagwara, under the guidance of our
    faculty mentor. All the information furnished in this report is based on our own work and is
    genuine.</p>
    <p>No part of this work has been submitted for any other degree/diploma at any university or
    institution.</p>
    <table class="sign" style="margin-top:5mm;">
      <tr><th></th><th>Student 1</th><th>Student 2</th><th>Student 3</th></tr>
      <tr><td><b>Name</b></td><td>Gaurav Rajbhar</td><td>P Arun Prashath</td><td>Nitesh Tiwari</td></tr>
      <tr><td><b>Registration No.</b></td><td>12417022</td><td>12418280</td><td>12418961</td></tr>
      <tr><td><b>Signature</b></td><td style="height:10mm;"></td><td></td><td></td></tr>
      <tr><td><b>Date</b></td><td></td><td></td><td></td></tr>
    </table>

    <h1 style="margin-top:7mm;">TABLE OF CONTENTS</h1>
    <table class="toc">
      <tr><td>1. &nbsp;Abstract</td><td class="dots"></td><td class="num">3</td></tr>
      <tr><td>2. &nbsp;Introduction and Objectives</td><td class="dots"></td><td class="num">3</td></tr>
      <tr><td>3. &nbsp;Dataset</td><td class="dots"></td><td class="num">3</td></tr>
      <tr><td>4. &nbsp;Methodology</td><td class="dots"></td><td class="num">3</td></tr>
      <tr><td>5. &nbsp;Implementation</td><td class="dots"></td><td class="num">4</td></tr>
      <tr><td>6. &nbsp;Results and Discussion</td><td class="dots"></td><td class="num">4</td></tr>
      <tr><td>7. &nbsp;Output and Demonstration</td><td class="dots"></td><td class="num">4</td></tr>
      <tr><td>8. &nbsp;Conclusion and Future Scope</td><td class="dots"></td><td class="num">4</td></tr>
      <tr><td>9. &nbsp;References</td><td class="dots"></td><td class="num">4</td></tr>
    </table>
    """

    body_1 = """
    <h2>1. Abstract</h2>
    <p>Brain tumours are among the most time-critical findings in medical imaging, and outlining them
    slice by slice is slow and observer dependent. This project builds a deep learning system that
    takes one axial brain MRI slice and produces a pixel-level tumour mask with a confidence score,
    presented in an interactive web application. The model is a UNet with a ResNet-34 encoder, scSE
    attention in the decoder, and a 2.5D input that stacks three neighbouring slices into nine
    channels, trained with a hybrid Tversky and focal loss on the Kaggle TCGA-LGG dataset (110
    patients, 7,858 image&ndash;mask pairs at 256&times;256). Quality is reported with slice Dice and
    IoU against the radiologist's ground truth, and with per-patient 3D Dice and volume error.</p>

    <h2>2. Introduction and Objectives</h2>
    <p>Manual delineation in a volumetric MRI study takes a radiologist tens of minutes and varies
    between observers. A convolutional model can produce a candidate mask in under a second, giving
    clinicians a starting point to correct rather than draw from scratch.</p>
    <p><b>Objectives:</b></p>
    <ul>
      <li>Build a UNet segmentation pipeline for axial brain MRI slices.</li>
      <li>Use 2.5D context without the cost of full 3D convolutions.</li>
      <li>Quantify quality with Dice and IoU, and volumetrically with 3D Dice and volume error.</li>
      <li>Deliver an interactive application that runs locally or hosted online.</li>
    </ul>

    <h2>3. Dataset</h2>
    <table>
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Dataset</td><td>Kaggle LGG MRI Segmentation (TCGA-LGG)</td></tr>
      <tr><td>Patients / image pairs</td><td>110 / 7,858</td></tr>
      <tr><td>Resolution and labels</td><td>256&times;256 grayscale; binary tumour mask per slice</td></tr>
      <tr><td>Splitting strategy</td><td>Grouped by patient &mdash; slices of one brain never straddle splits</td></tr>
    </table>

    <h2>4. Methodology</h2>
    <p><b>Architecture.</b> A UNet from <code>segmentation-models-pytorch</code>. The ResNet-34
    encoder, initialised from ImageNet, compresses the slice into multi-scale features; the decoder
    rebuilds the mask using skip connections. scSE attention re-weights channels and spatial regions
    in the decoder, which helps because the tumour occupies a small fraction of the slice. The
    ResNet-34 model holds 24.6 million parameters.</p>
    <p><b>2.5D input.</b> Each sample is three consecutive axial slices tiled into nine channels, so
    the network has local context while remaining a 2D model. Intensities are clipped to the
    1st&ndash;99th percentile and kept on the 0&ndash;255 scale; normalising to 0&ndash;1 was found to
    collapse the saved checkpoints' output to a constant.</p>
    <p><b>Loss and training.</b> A hybrid of Tversky loss (alpha 0.3, beta 0.7) and focal loss
    (alpha 0.75, gamma 2.0) addresses the severe class imbalance. Optimisation uses AdamW with a lower
    encoder learning rate (1e-4) than the decoder (1e-3), warm-up followed by cosine decay, and mixed
    precision.</p>
    """

    body_2 = """
    <h2>5. Implementation</h2>
    <p>The system has four parts: trained checkpoints, a Streamlit front end, a training script and a
    Colab notebook. The front end rebuilds each architecture from the run's <code>config.json</code>,
    reads the channel count from the encoder stem and loads weights with <code>strict=True</code>, so
    the demo network is exactly the trained network. Three checkpoint sets ship: ResNet-34 2.5D (folds
    0&ndash;3), EfficientNet-B4 2.5D (folds 0&ndash;1) and an upgraded EfficientNet-B4 (fold 0). The
    training script uses grouped K-fold validation over patients, sweeps the threshold on each fold's
    validation split only, and accumulates 3D Dice and volume error in the same pass.</p>

    <h2>6. Results and Discussion</h2>
    <table>
      <tr><th>Metric (ResNet-34, threshold 0.325)</th><th>Value</th><th>Interpretation</th></tr>
      <tr><td>Test Dice, tumour slices</td><td>0.674</td><td>Reasonable slice-level overlap</td></tr>
      <tr><td>Test Dice, all slices</td><td>0.535</td><td>Includes slices with no tumour</td></tr>
      <tr><td>Test IoU, tumour slices</td><td>0.559</td><td>Stricter overlap measure</td></tr>
      <tr><td>Out-of-fold validation Dice</td><td>0.481</td><td>Honest generalisation estimate</td></tr>
      <tr><td>3D Dice per patient</td><td>0.482</td><td>Volumetric agreement</td></tr>
      <tr><td>Median volume error</td><td>119.9%</td><td>Predicted volumes roughly double the truth</td></tr>
    </table>
    <p>Two weaknesses are visible: the model systematically <b>over-segments</b>, and the headline
    threshold was selected against the test split itself, so 0.674 is optimistic while the out-of-fold
    0.481 is the defensible number.</p>
    <div class="note"><p><b>Reproducibility note.</b> Re-evaluated against the dataset masks, the saved
    checkpoints reached only about 0.10 Dice. This held across four 9-channel encodings, five
    normalisations, two input sizes, thresholds 0.1&ndash;0.8 and all eight orientations. The weights
    load strictly, so the architecture matches; they do not reproduce the recorded scores on this data.
    The app states this openly and the training script exists to retrain defensible models.</p></div>

    <h2>7. Output and Demonstration</h2>
    <div class="fig"><img src="improved_r34_25d-20261115T150005Z-1-001/improved_r34_25d/overlay_best.png"
      style="width:89%;"></div>
    <div class="cap">Figure 1: Predictions on held-out slices &mdash; input, ground truth, probability
    map and overlay for best, random and worst cases.</div>
    <div class="fig"><img src="figure_app.png" style="width:89%;"></div>
    <div class="cap">Figure 2: Live application output for one slice &mdash; MRI input, ground truth,
    probability map, and overlap (green = truth, red = prediction).</div>
    <p><b>Application views.</b> The interface shows these four panels for any slice, names the model in
    use (UNet, ResNet-34 encoder, scSE attention, 9-channel 2.5D input, 24.6 M parameters, exact
    checkpoint file) and explains each number: Dice and IoU for overlap, predicted versus ground-truth area for
    over-segmentation, and mean confidence for certainty. Users may upload
    their own slices or evaluate every tumour slice of a patient in one click.</p>

    <h2>8. Conclusion and Future Scope</h2>
    <p>A complete, runnable system was built: a 2.5D UNet trained on TCGA-LGG, a web application that
    explains every output it produces, and a reproducible training pipeline. The investigation also
    showed the shipped checkpoints do not reproduce their recorded scores, so retraining is the next
    step &mdash; with the provided script, out-of-fold threshold selection, over-segmentation
    suppression, and voxel spacing recorded for mm&sup3; volumes.</p>

    <h2>9. References</h2>
    <ol>
      <li>Ronneberger et al., <i>U-Net</i>, MICCAI 2015.</li>
      <li>He et al., <i>Deep Residual Learning for Image Recognition</i>, CVPR 2016.</li>
      <li>Roy et al., <i>scSE in Fully Convolutional Networks</i>, MICCAI 2018.</li>
      <li>Kaggle, <i>LGG MRI Segmentation</i> dataset (TCGA-LGG), 2019.</li>
    </ol>
    """

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
    {page(cover, 1)}{page(declaration, 2)}{page(body_1, 3)}{page(body_2, 4)}
    </body></html>"""


def main() -> None:
    html = build_html()
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


if __name__ == "__main__":
    main()
