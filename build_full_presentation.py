import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette
    BG_DARK = RGBColor(15, 23, 42)        # Slate 900
    TEXT_LIGHT = RGBColor(248, 250, 252)  # Slate 50
    TEXT_MUTED = RGBColor(148, 163, 184)  # Slate 400
    CARD_BG = RGBColor(30, 41, 59)        # Slate 800
    CARD_BORDER = RGBColor(51, 65, 85)    # Slate 700
    TABLE_HDR_BG = RGBColor(30, 41, 59)   # Slate 800
    TABLE_ROW_ALT = RGBColor(24, 32, 47)  # Slate 850
    PRIMARY = RGBColor(99, 102, 241)      # Indigo 500
    ACCENT = RGBColor(20, 184, 166)       # Teal 500
    ACCENT_ALT = RGBColor(244, 63, 94)    # Rose 500
    AMBER = RGBColor(245, 158, 11)        # Amber 500
    WHITE = RGBColor(255, 255, 255)

    blank_layout = prs.slide_layouts[6]

    def set_bg(slide, color=BG_DARK):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_text="MAJOR PROJECT (30% COMPLETION REVIEW)"):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT
        p_cat.font.name = "Arial"

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_LIGHT
        p_title.font.name = "Arial"

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1)

    dec_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(0.15), Inches(4.5))
    dec_bar.fill.solid()
    dec_bar.fill.fore_color.rgb = PRIMARY
    dec_bar.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11.2), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "COLLEGE MAJOR PROJECT — PHASE 1 REVIEW (30% COMPLETION)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT
    p.space_after = Pt(14)

    p = tf1.add_paragraph()
    p.text = "Compound Facial Emotion Recognition using Morphological Splicing & Multi-Branch Attention-Transformer Networks"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.space_after = Pt(18)

    p = tf1.add_paragraph()
    p.text = "A Deep Learning Framework combining SA-CNN, ALSTM, and Vision Transformers (ViT) with Dynamic Attention Fusion for 11 Compound Expression Classes"
    p.font.size = Pt(14)
    p.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(30)

    info_box = s1.shapes.add_textbox(Inches(1.2), Inches(5.8), Inches(11.2), Inches(1.0))
    tf_info = info_box.text_frame
    p_info = tf_info.paragraphs[0]
    p_info.text = "Domain: Deep Learning & Computer Vision  |  Progress Stage: 30% Work Completed  |  Department of Computer Science & Engineering"
    p_info.font.size = Pt(12)
    p_info.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 2: Introduction & Motivation
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2)
    add_header(s2, "Introduction & Motivation: Beyond Basic Emotions")

    card_w = Inches(3.64)
    card_h = Inches(4.8)
    gap = Inches(0.4)
    y_pos = Inches(1.8)

    cdata2 = [
        ("Beyond Basic Emotions", "Traditional FER systems categorize faces into 6-7 basic emotions (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral).\n\nReal-world human expressions are subtle and blended, giving rise to Compound Emotions (e.g., 'Happily Surprised', 'Fearfully Angry').", ACCENT),
        ("Key Technical Challenges", "• Micro-Expression Subtleties: Overlapping facial muscle movements across blended emotion pairs.\n• Intra-Class Ambiguity: High visual similarity between classes like 'Fearfully Angry' and 'Sadly Surprised'.\n• Feature Granularity: Requires local spatial, sequential, and global context.", PRIMARY),
        ("Practical Applications", "• Driver State & Mental Health Monitoring\n• AI-driven HCI & Virtual Assistant Empathy\n• Online Learning Student Engagement Tracking\n• Automated Customer Feedback & Behavior Analysis", ACCENT_ALT)
    ]

    for i, (ctitle, cdesc, ccolor) in enumerate(cdata2):
        x = Inches(0.8) + i * (card_w + gap)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y_pos, card_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER

        tb = s2.shapes.add_textbox(x + Inches(0.2), y_pos + Inches(0.2), card_w - Inches(0.4), card_h - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = ctitle
        pt.font.size = Pt(17)
        pt.font.bold = True
        pt.font.color.rgb = ccolor
        pt.space_after = Pt(12)

        pd = tf.add_paragraph()
        pd.text = cdesc
        pd.font.size = Pt(13)
        pd.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 3: Problem Statement & Objectives
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3)
    add_header(s3, "Problem Statement & Major Project Objectives")

    w3 = Inches(5.6)
    h3 = Inches(4.8)

    # Problem Box
    b_prob = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), w3, h3)
    b_prob.fill.solid()
    b_prob.fill.fore_color.rgb = CARD_BG
    b_prob.line.color.rgb = CARD_BORDER

    tb_p = s3.shapes.add_textbox(Inches(1.0), Inches(2.0), w3 - Inches(0.4), h3 - Inches(0.4))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    p = tf_p.paragraphs[0]
    p.text = "PROBLEM STATEMENT"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_ALT
    p.space_after = Pt(10)

    p = tf_p.add_paragraph()
    p.text = "Existing Facial Emotion Recognition (FER) models rely heavily on single-stream CNNs or standard Transformers, which fail to capture fine-grained compound emotion nuances.\n\nThere is a critical need for a multi-branch architecture that combines spatial attention, sequential relationship modeling, and global vision transformers to accurately classify 11 compound emotions while maintaining computational efficiency."
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_LIGHT

    # Objectives Box
    b_obj = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), w3, h3)
    b_obj.fill.solid()
    b_obj.fill.fore_color.rgb = CARD_BG
    b_obj.line.color.rgb = CARD_BORDER

    tb_o = s3.shapes.add_textbox(Inches(7.0), Inches(2.0), w3 - Inches(0.4), h3 - Inches(0.4))
    tf_o = tb_o.text_frame
    tf_o.word_wrap = True
    p = tf_o.paragraphs[0]
    p.text = "PROJECT OBJECTIVES"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT
    p.space_after = Pt(10)

    objs = [
        "1. Develop Morphological Splicing data augmentation pipeline for expression feature enhancement.",
        "2. Implement a Multi-Branch Feature Extractor (SA-CNN, ALSTM, and Vision Transformer).",
        "3. Design Generic Attention Fusion to dynamically balance local spatial and global contextual embeddings.",
        "4. Conduct comprehensive empirical benchmarking across 6 baseline/hybrid architectures on RAF-DB, AffectNet, CK+, and JAFFE.",
        "5. Achieve real-time inference capability (>40 FPS) with low parameter footprint."
    ]
    for o in objs:
        p = tf_o.add_paragraph()
        p.text = o
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_LIGHT
        p.space_after = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 4: Literature Survey & Gap Analysis
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4)
    add_header(s4, "Literature Survey & Research Gap Analysis")

    rows = 4
    cols = 4
    left = Inches(0.8)
    top = Inches(1.8)
    width = Inches(11.733)
    height = Inches(4.8)

    table_shape = s4.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(2.3)
    table.columns[1].width = Inches(2.8)
    table.columns[2].width = Inches(3.2)
    table.columns[3].width = Inches(3.433)

    headers = ["Approach / Model Class", "Key Advantages", "Identified Limitations", "Our Proposed Solution"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = WHITE

    data4 = [
        ("Traditional CNNs\n(ResNet, VGG)", "Good local spatial feature extraction", "Fails to capture global facial context and long-range patch dependencies", "Integrate Vision Transformer (ViT) backbone for global patch self-attention"),
        ("Pure Vision Transformers\n(ViT-Base)", "Excellent global attention modeling", "High computational FLOPs; misses fine-grained local muscle contractions", "Combine Spatial Attention CNN (SA-CNN) for fine facial muscle features"),
        ("RNN / LSTM models", "Captures sequential & spatial region correlations", "Prone to gradient issues and lacks dynamic attention fusion", "Incorporate Spatial Attention LSTM (ALSTM) fused via Dynamic Gated Softmax")
    ]

    for row_idx, row_data in enumerate(data4, start=1):
        bg_c = TABLE_HDR_BG if row_idx % 2 == 1 else TABLE_ROW_ALT
        for col_idx, text in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_c
            p = cell.text_frame.paragraphs[0]
            p.text = text
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 5: Proposed Architecture & System Design
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5)
    add_header(s5, "Proposed Architecture: End-to-End System Design")

    # 4 horizontal boxes showing pipeline
    bw = Inches(2.6)
    bh = Inches(4.8)
    bgap = Inches(0.4)
    by = Inches(1.8)

    pipeline_steps = [
        ("Stage 1: Input & Splicing", "• Input Face Image (224x224)\n• Morphological Splicing\n• Normalization & Augmentation\n• Multi-scale facial cropping", ACCENT),
        ("Stage 2: Feature Extractors", "• SA-CNN: Spatial & Channel Attention\n• ALSTM: Sequential Region Attention\n• ViT: Patch Self-Attention (16x16)", PRIMARY),
        ("Stage 3: Attention Fusion", "• GenericAttentionFusion\n• Gated Softmax Weighting\n• Concatenation -> 128 -> Softmax\n• 512-dim Fused Embedding", ACCENT_ALT),
        ("Stage 4: MLP Classification", "• Linear (512 -> 256)\n• LayerNorm + GELU + Dropout\n• Linear (256 -> 11)\n• 11 Compound Emotion Probabilities", AMBER)
    ]

    for i, (stitle, sdesc, scolor) in enumerate(pipeline_steps):
        bx = Inches(0.8) + i * (bw + bgap)
        b = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, bw, bh)
        b.fill.solid()
        b.fill.fore_color.rgb = CARD_BG
        b.line.color.rgb = CARD_BORDER

        tb = s5.shapes.add_textbox(bx + Inches(0.15), by + Inches(0.2), bw - Inches(0.3), bh - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True

        pt = tf.paragraphs[0]
        pt.text = stitle
        pt.font.size = Pt(15)
        pt.font.bold = True
        pt.font.color.rgb = scolor
        pt.space_after = Pt(14)

        pd = tf.add_paragraph()
        pd.text = sdesc
        pd.font.size = Pt(12)
        pd.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 6: Datasets & Experimental Setup
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6)
    add_header(s6, "Datasets & Experimental Setup")

    w6 = Inches(5.6)
    h6 = Inches(4.8)

    # Box 1: Datasets & Classes
    b1 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), w6, h6)
    b1.fill.solid()
    b1.fill.fore_color.rgb = CARD_BG
    b1.line.color.rgb = CARD_BORDER

    tb1 = s6.shapes.add_textbox(Inches(1.0), Inches(2.0), w6 - Inches(0.4), h6 - Inches(0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "11 COMPOUND EMOTION CLASSES & DATASETS"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT
    p.space_after = Pt(10)

    p = tf1.add_paragraph()
    p.text = "Compound Classes (11):\nHappily Surprised, Happily Disgusted, Sadly Fearful, Sadly Angry, Sadly Surprised, Sadly Disgusted, Fearfully Angry, Fearfully Surprised, Angrily Surprised, Angrily Disgusted, Disgustedly Surprised.\n\nDatasets Evaluated:\n• RAF-DB (Real-world Affective Faces - 3,068 test split)\n• AffectNet, JAFFE, CK+, FER-2013"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    # Box 2: Hyperparameters & Curriculum
    b2 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), w6, h6)
    b2.fill.solid()
    b2.fill.fore_color.rgb = CARD_BG
    b2.line.color.rgb = CARD_BORDER

    tb2 = s6.shapes.add_textbox(Inches(7.0), Inches(2.0), w6 - Inches(0.4), h6 - Inches(0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "TRAINING HYPERPARAMETERS & CURRICULUM"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = PRIMARY
    p.space_after = Pt(10)

    p = tf2.add_paragraph()
    p.text = "• Image Resolution: 224 x 224 | Batch Size: 32\n• Optimizer: AdamW (lr = 2e-4, weight_decay = 1e-4)\n• Learning Rate Schedule: OneCycleLR\n• Embedding Dim: 512 | LSTM Hidden: 256 (2 layers)\n• ViT Backbone: vit_base_patch16_224\n\n3-Stage Curriculum Training:\n1. Stage 1: Backbone Frozen (train head & fusion)\n2. Stage 2: Partial Unfreeze (fine-tune last n blocks)\n3. Stage 3: Full Unfreeze (end-to-end optimization)"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 7: 30% Completion Status Breakdown
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7)
    add_header(s7, "Phase 1 Progress: 30% Project Completion Status")

    # Big Banner
    ban = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.733), Inches(0.8))
    ban.fill.solid()
    ban.fill.fore_color.rgb = PRIMARY
    ban.line.fill.background()

    tb_ban = s7.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(0.6))
    tf_ban = tb_ban.text_frame
    p = tf_ban.paragraphs[0]
    p.text = "STATUS: PHASE 1 COMPLETED (30% MILESTONE ACHIEVED) — Core Pipeline & Baseline Evaluation Ready"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = WHITE

    # Completed vs Planned Table
    rows = 6
    cols = 3
    t_shape = s7.shapes.add_table(rows, cols, Inches(0.8), Inches(2.8), Inches(11.733), Inches(3.8))
    table = t_shape.table
    table.columns[0].width = Inches(4.5)
    table.columns[1].width = Inches(2.2)
    table.columns[2].width = Inches(5.033)

    h_7 = ["Project Module / Task", "Completion Status", "Key Deliverable / Outcome"]
    for i, h in enumerate(h_7):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT

    tasks7 = [
        ("Architecture Design & Module Construction", "100% (Done)", "SA-CNN, ALSTM, ViT, and Generic Attention Fusion built in PyTorch"),
        ("Data Pipeline & Morphological Splicing", "100% (Done)", "RAF-DB, AffectNet, CK+, JAFFE dataloaders & splicing augmentation"),
        ("Multi-Model Benchmark Suite (6 Models)", "100% (Done)", "ConvNeXt+ViT, SACNN+ViT, EfficientNetV2+ViT, Proposed Model, etc."),
        ("Evaluation Pipeline & Metric Engine", "100% (Done)", "Accuracy, Precision, Recall, F1, FLOPs, Params, FPS, Confusion Matrix"),
        ("Loss Optimization & Real-Time GUI (Phases 2-3)", "0% (Planned)", "Scheduled for 70% and 100% completion reviews")
    ]

    for row_idx, (t_name, t_stat, t_out) in enumerate(tasks7, start=1):
        bg_c = TABLE_HDR_BG if row_idx % 2 == 1 else TABLE_ROW_ALT
        c0 = table.cell(row_idx, 0)
        c1 = table.cell(row_idx, 1)
        c2 = table.cell(row_idx, 2)
        c0.fill.solid(); c0.fill.fore_color.rgb = bg_c
        c1.fill.solid(); c1.fill.fore_color.rgb = bg_c
        c2.fill.solid(); c2.fill.fore_color.rgb = bg_c

        p0 = c0.text_frame.paragraphs[0]; p0.text = t_name; p0.font.size = Pt(11); p0.font.color.rgb = TEXT_LIGHT
        p1 = c1.text_frame.paragraphs[0]; p1.text = t_stat; p1.font.size = Pt(11); p1.font.bold = True; p1.font.color.rgb = ACCENT if "100%" in t_stat else AMBER
        p2 = c2.text_frame.paragraphs[0]; p2.text = t_out; p2.font.size = Pt(11); p2.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 8: Experimental Benchmarking & Results (30% Milestone)
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8)
    add_header(s8, "30% Benchmark Results: Performance Across 6 Architectures")

    rows = 7
    cols = 8
    t_shape8 = s8.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8))
    table8 = t_shape8.table

    col_w = [Inches(2.533), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.4)]
    for i, w in enumerate(col_w):
        table8.columns[i].width = w

    h_8 = ["Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "Params (M)", "FLOPs (G)", "FPS (Speed)"]
    for i, h in enumerate(h_8):
        cell = table8.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = WHITE

    res_data = [
        ("ConvNeXt + ViT", "87.97%", "82.89%", "78.74%", "80.43%", "114.9M", "42.6G", "51.0 FPS"),
        ("SA-CNN + ViT", "87.61%", "81.15%", "78.89%", "79.81%", "103.1M", "44.3G", "53.1 FPS"),
        ("EfficientNetV2 + ViT", "87.35%", "82.45%", "77.74%", "79.65%", "107.6M", "39.1G", "40.9 FPS"),
        ("Proposed (Splicing+SACNN+ALSTM+ViT)", "86.86%", "80.92%", "77.18%", "78.71%", "106.4M", "45.5G", "43.4 FPS"),
        ("ALSTM + ViT", "86.86%", "81.90%", "77.36%", "79.16%", "89.9M", "34.9G", "53.4 FPS"),
        ("SA-CNN + ALSTM", "80.67%", "63.24%", "62.03%", "62.38%", "20.1M", "11.8G", "79.8 FPS")
    ]

    for row_idx, rdata in enumerate(res_data, start=1):
        bg_c = TABLE_HDR_BG if row_idx % 2 == 1 else TABLE_ROW_ALT
        is_best = (row_idx == 1)
        is_light = (row_idx == 6)

        for col_idx, text in enumerate(rdata):
            cell = table8.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_c
            p = cell.text_frame.paragraphs[0]
            p.text = text
            p.font.size = Pt(10)
            if is_best and col_idx in [0, 1]:
                p.font.bold = True
                p.font.color.rgb = ACCENT
            elif is_light and col_idx in [5, 7]:
                p.font.bold = True
                p.font.color.rgb = AMBER
            else:
                p.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 9: Architectural Trade-Off Analysis
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9)
    add_header(s9, "Key Findings: Performance vs Computational Complexity")

    c_w = Inches(5.6)
    c_h = Inches(4.8)

    # Left Card: ViT Backbones
    b_l = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), c_w, c_h)
    b_l.fill.solid(); b_l.fill.fore_color.rgb = CARD_BG; b_l.line.color.rgb = CARD_BORDER

    tb_l = s9.shapes.add_textbox(Inches(1.0), Inches(2.0), c_w - Inches(0.4), c_h - Inches(0.4))
    tf_l = tb_l.text_frame; tf_l.word_wrap = True
    p = tf_l.paragraphs[0]; p.text = "HIGH-ACCURACY TRANSFORMER VARIANTS"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = ACCENT; p.space_after = Pt(10)

    p = tf_l.add_paragraph()
    p.text = "• ConvNeXt + ViT achieved peak accuracy of 87.97% with macro F1 of 80.43%.\n• SA-CNN + ViT achieved 87.61% accuracy with excellent spatial feature representation.\n• Vision Transformer patch self-attention is essential for disambiguating compound facial expressions.\n• Inference speeds remain above 50 FPS on GPU."
    p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT

    # Right Card: Lightweight Variants
    b_r = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), c_w, c_h)
    b_r.fill.solid(); b_r.fill.fore_color.rgb = CARD_BG; b_r.line.color.rgb = CARD_BORDER

    tb_r = s9.shapes.add_textbox(Inches(7.0), Inches(2.0), c_w - Inches(0.4), c_h - Inches(0.4))
    tf_r = tb_r.text_frame; tf_r.word_wrap = True
    p = tf_r.paragraphs[0]; p.text = "LIGHTWEIGHT & REAL-TIME EDGE VARIANTS"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = AMBER; p.space_after = Pt(10)

    p = tf_r.add_paragraph()
    p.text = "• SA-CNN + ALSTM operates with only 20.1M parameters and 11.8 GFLOPs.\n• Reaches ultra-fast 79.8 FPS inference, making it ideal for edge devices and mobile systems.\n• Trade-off: Accuracy drops to 80.67%, proving that Transformer self-attention is critical for compound expression nuances.\n• Fused models strike an optimal balance for server-side real-time deployment."
    p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 10: Error Analysis & Confusion Matrix Insights
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10)
    add_header(s10, "Error Analysis & Fine-Grained Expression Confusion")

    c10_w = Inches(3.64)
    c10_h = Inches(4.8)

    cdata10 = [
        ("Primary Confusion Pair", "Across ALL models, the most confused compound emotion pair was:\n\n'Fearfully Angry' <-> 'Sadly Surprised'\n(36 to 61 misclassified samples)", ACCENT_ALT),
        ("Root Cause Analysis", "• Morphological Overlap: Both expressions share contracted brow muscles (corrugator supercilii) and widened eyes.\n• Imbalanced Split: Subtle micro-expression differences require adaptive loss re-weighting.", PRIMARY),
        ("Mitigation in Phase 2", "• Implement Class-Weighted Cross-Entropy Loss.\n• Introduce Focal Loss to heavily penalize hard compound emotion misclassifications.\n• Fine-tune attention gates specifically on confused class boundaries.", ACCENT)
    ]

    for i, (ctitle, cdesc, ccolor) in enumerate(cdata10):
        x = Inches(0.8) + i * (c10_w + gap)
        card = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y_pos, c10_w, c10_h)
        card.fill.solid(); card.fill.fore_color.rgb = CARD_BG; card.line.color.rgb = CARD_BORDER

        tb = s10.shapes.add_textbox(x + Inches(0.2), y_pos + Inches(0.2), c10_w - Inches(0.4), c10_h - Inches(0.4))
        tf = tb.text_frame; tf.word_wrap = True

        pt = tf.paragraphs[0]; pt.text = ctitle; pt.font.size = Pt(17); pt.font.bold = True; pt.font.color.rgb = ccolor; pt.space_after = Pt(12)
        pd = tf.add_paragraph(); pd.text = cdesc; pd.font.size = Pt(13); pd.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 11: Remaining Work Plan (70% Roadmap)
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    set_bg(s11)
    add_header(s11, "Roadmap: Plan for Remaining 70% Project Completion")

    w11 = Inches(5.6)
    h11 = Inches(4.8)

    # Phase 2 Card
    b_p2 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), w11, h11)
    b_p2.fill.solid(); b_p2.fill.fore_color.rgb = CARD_BG; b_p2.line.color.rgb = CARD_BORDER

    tb_p2 = s11.shapes.add_textbox(Inches(1.0), Inches(2.0), w11 - Inches(0.4), h11 - Inches(0.4))
    tf_p2 = tb_p2.text_frame; tf_p2.word_wrap = True
    p = tf_p2.paragraphs[0]; p.text = "PHASE 2: OPTIMIZATION & EXTENSION (30% -> 70%)"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = ACCENT; p.space_after = Pt(10)

    p2_items = [
        "• Implement Focal Loss & Class-Weighted Cross-Entropy to fix 'Fearfully Angry' vs 'Sadly Surprised' confusion.",
        "• Execute cross-dataset generalization experiments on full AffectNet and FER-2013 sets.",
        "• Conduct ablation studies on Morphological Splicing parameters.",
        "• Apply INT8 Quantization and TensorRT / ONNX export for edge acceleration."
    ]
    for item in p2_items:
        p = tf_p2.add_paragraph(); p.text = item; p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT; p.space_after = Pt(8)

    # Phase 3 Card
    b_p3 = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), w11, h11)
    b_p3.fill.solid(); b_p3.fill.fore_color.rgb = CARD_BG; b_p3.line.color.rgb = CARD_BORDER

    tb_p3 = s11.shapes.add_textbox(Inches(7.0), Inches(2.0), w11 - Inches(0.4), h11 - Inches(0.4))
    tf_p3 = tb_p3.text_frame; tf_p3.word_wrap = True
    p = tf_p3.paragraphs[0]; p.text = "PHASE 3: DEPLOYMENT & DOCUMENTATION (70% -> 100%)"; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = PRIMARY; p.space_after = Pt(10)

    p3_items = [
        "• Build real-time interactive Web Application / Desktop GUI for webcam video stream emotion prediction.",
        "• Integrate live emotion probability radar charts and logging.",
        "• Finalize Major Project Thesis & Documentation report.",
        "• Prepare research paper manuscript for publication submission."
    ]
    for item in p3_items:
        p = tf_p3.add_paragraph(); p.text = item; p.font.size = Pt(12); p.font.color.rgb = TEXT_LIGHT; p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 12: Project Implementation Timeline
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    set_bg(s12)
    add_header(s12, "Project Timeline & Milestone Schedule")

    rows = 5
    cols = 4
    t_shape12 = s12.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8))
    table12 = t_shape12.table
    table12.columns[0].width = Inches(2.2)
    table12.columns[1].width = Inches(4.5)
    table12.columns[2].width = Inches(2.3)
    table12.columns[3].width = Inches(2.733)

    h_12 = ["Phase / Timeframe", "Key Milestones & Deliverables", "Target Completion", "Status"]
    for i, h in enumerate(h_12):
        cell = table12.cell(0, i)
        cell.fill.solid(); cell.fill.fore_color.rgb = PRIMARY
        p = cell.text_frame.paragraphs[0]; p.text = h; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = WHITE

    tdata12 = [
        ("Phase 1 (Months 1-2)", "Literature review, dataset setup, multi-branch architecture design, baseline evaluation suite", "30% Milestone", "COMPLETED ✅"),
        ("Phase 2A (Month 3)", "Focal loss optimization, hyperparameter tuning, ablation experiments", "50% Milestone", "IN PROGRESS ⏳"),
        ("Phase 2B (Month 4)", "Cross-dataset validation, model quantization (ONNX/TensorRT)", "70% Milestone", "PLANNED 📅"),
        ("Phase 3 (Months 5-6)", "Real-time GUI web app, final thesis report, viva presentation & paper submission", "100% Final Review", "PLANNED 📅")
    ]

    for row_idx, rdata in enumerate(tdata12, start=1):
        bg_c = TABLE_HDR_BG if row_idx % 2 == 1 else TABLE_ROW_ALT
        for col_idx, text in enumerate(rdata):
            cell = table12.cell(row_idx, col_idx)
            cell.fill.solid(); cell.fill.fore_color.rgb = bg_c
            p = cell.text_frame.paragraphs[0]; p.text = text; p.font.size = Pt(11)
            if col_idx == 3:
                p.font.bold = True
                p.font.color.rgb = ACCENT if "COMPLETED" in text else AMBER
            else:
                p.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 13: Expected Deliverables & Impact
    # -------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    set_bg(s13)
    add_header(s13, "Expected Major Project Deliverables & Key Impact")

    c13_w = Inches(3.64)
    c13_h = Inches(4.8)

    cdata13 = [
        ("Trained Models & Suite", "• 6 Fully-Restored & Evaluated Model Checkpoints.\n• PyTorch implementation of Morphological Splicing & Dynamic Attention Fusion.", ACCENT),
        ("Empirical Study & Code", "• Comprehensive benchmarking dataset comparing Params, FLOPs, Accuracy, and Real-Time FPS.\n• Complete automated evaluation pipeline.", PRIMARY),
        ("Real-World Deployment", "• Web-based Real-Time Compound FER GUI Application.\n• Published Research Paper & Major Project Thesis Documentation.", ACCENT_ALT)
    ]

    for i, (ctitle, cdesc, ccolor) in enumerate(cdata13):
        x = Inches(0.8) + i * (c13_w + gap)
        card = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y_pos, c13_w, c13_h)
        card.fill.solid(); card.fill.fore_color.rgb = CARD_BG; card.line.color.rgb = CARD_BORDER

        tb = s13.shapes.add_textbox(x + Inches(0.2), y_pos + Inches(0.2), c13_w - Inches(0.4), c13_h - Inches(0.4))
        tf = tb.text_frame; tf.word_wrap = True

        pt = tf.paragraphs[0]; pt.text = ctitle; pt.font.size = Pt(17); pt.font.bold = True; pt.font.color.rgb = ccolor; pt.space_after = Pt(12)
        pd = tf.add_paragraph(); pd.text = cdesc; pd.font.size = Pt(13); pd.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 14: Conclusion & Q&A
    # -------------------------------------------------------------
    s14 = prs.slides.add_slide(blank_layout)
    set_bg(s14)
    add_header(s14, "Conclusion & Q&A")

    card_c = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(1.8), Inches(10.333), Inches(4.8))
    card_c.fill.solid(); card_c.fill.fore_color.rgb = CARD_BG; card_c.line.color.rgb = CARD_BORDER

    tb_c = s14.shapes.add_textbox(Inches(1.8), Inches(2.1), Inches(9.733), Inches(4.2))
    tf_c = tb_c.text_frame; tf_c.word_wrap = True

    p = tf_c.paragraphs[0]
    p.text = "SUMMARY OF PHASE 1 (30% COMPLETION)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = ACCENT; p.space_after = Pt(12)

    p = tf_c.add_paragraph()
    p.text = "• Successfully built and evaluated 6 deep learning hybrid models for 11 compound facial emotion classes.\n• Demonstrated top accuracy of 87.97% (ConvNeXt + ViT) and 87.61% (SA-CNN + ViT).\n• Identified lightweight real-time variant (SA-CNN + ALSTM @ 79.8 FPS).\n• Solved core pipeline requirements and laid strong technical groundwork for Phase 2 optimization."
    p.font.size = Pt(13); p.font.color.rgb = TEXT_LIGHT; p.space_after = Pt(24)

    p = tf_c.add_paragraph()
    p.text = "THANK YOU!\nOpen for Questions, Suggestions, and Faculty Feedback."
    p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER

    # Save presentation
    output_path = "/Users/akshaym/Desktop/Major project/Compound_Facial_Emotion_Recognition_30Percent_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

create_deck()
