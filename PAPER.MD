Overview
The paper “Identity Document to Selfie Face Matching Across Adolescence” addresses the challenge of verifying a person’s identity by matching a “selfie” (live face image) to an image on an ID document when the two images are separated by several years of adolescent growth. This problem is particularly difficult because facial appearance can change substantially during adolescence. The study introduces a large-scale private dataset (the Chilean Young Adult or CHIYA dataset), evaluates existing face‐matching systems (both off‐the‐shelf and open‐source), and proposes a fine‐tuning methodology based on triplet loss to significantly improve accuracy for this specific task.  ￼ ￼

⸻

1. Problem Statement

Matching live “selfie” images to ID document photos is commonplace in applications such as online account opening or benefit enrollment. A particularly challenging scenario arises when the ID‐document photo was taken in early adolescence (as young as age 9) and the live selfie is captured in late adolescence (age 18–19). In such cases, natural facial growth can induce large intra‐subject variations that degrade conventional face matchers’ accuracy. Prior work has shown that even state‐of‐the‐art matchers like ArcFace struggle under these conditions, achieving true acceptance rates (TAR) well below acceptable thresholds at low false accept rates (FAR) for large age differences during adolescence.  ￼

⸻

2. CHIYA Dataset
	•	Composition: The CHIYA dataset consists of 263,490 subjects, each represented by exactly two images—a scanned national ID card photo (captured at ages 9–18) and a contemporaneous selfie (age 18–19). This totals 526,980 images.  ￼
	•	Image Acquisition: ID‐card images were obtained when individuals enrolled for a government benefit (2017 onward) in Chile. Selfies were collected using various mobile devices at the time of enrollment. Because the ID cards come in two formats (“yellow card” issued pre‐2013 and “blue card” issued post‐2013), preprocessing must handle both layouts.  ￼
	•	Age Gap Subsets: To analyze performance as a function of age gap, the authors define five subsets:
	•	i10s1819: ID at age 10 vs. selfie at age 18–19 (324 subjects)
	•	i12s1819: ID at age 12 vs. selfie at age 18–19 (1,182 subjects)
	•	i14s1819: ID at age 14 vs. selfie at age 18–19 (1,545 subjects)
	•	i16s1819: ID at age 16 vs. selfie at age 18–19 (919 subjects)
	•	i18s1819: ID at age 18 vs. selfie at age 18–19 (1,396 subjects)
This stratification highlights the increasing difficulty as the age difference grows.  ￼

⸻

3. Preprocessing Pipeline
	1.	ID‐Card Segmentation and Redaction
	•	SIFT keypoint detection and homography are used to segment each ID card, rectify its orientation, and warp it into a canonical flat view.
	•	Personally identifiable information (e.g., name, ID number) is redacted.
	•	An OCR engine extracts metadata (e.g., gender, birth year, ID‐issue year) to estimate each subject’s age at the time the ID image was created.  ￼
	2.	Face Detection and Alignment
	•	RetinaFace is applied to detect and align faces in both ID‐card images and live selfies.
	•	After alignment, faces are cropped and resized—224 × 224 pixels for most matchers, 112 × 112 (center‐cropped to 96 × 112) for DocFace+—depending on each matcher’s input requirements.  ￼
	3.	Quality Filtering
	•	Approximately 20,262 subjects without a detectable face in either the ID or selfie image are excluded, leaving 243,228 usable pairs.  ￼

⸻

4. Baseline Face Matchers Evaluated

The study evaluates a mix of commercial off‐the‐shelf (COTS), government off‐the‐shelf (GOTS), and open‐source deep‐learning matchers on the CHIYA dataset without fine‐tuning:
	1.	COTS‐1 (Commercial, proprietary)
	2.	GOTS‐1 and GOTS‐2 (Government‐developed, proprietary)
	3.	VGGFace [16] (Open‐source CNN trained on web‐scraped images)
	4.	FaceNet [19] (Open‐source CNN; input size 160 × 160)
	5.	VGGFace2 [5] (Open‐source CNN; variation of VGG)
	6.	ArcFace [7] (Open‐source CNN; input size 112 × 112)
	7.	DocFace+ Base Model [21] (Open‐source, specialized for ID‐to‐selfie matching; input size 112 × 112 with DWI optimization)

In initial experiments, ArcFace achieved the highest TAR among off‐the‐shelf matchers but still suffered significant drops in accuracy as the age gap widened (e.g., TAR ≈ 15–85% averaged over age gaps). DocFace+ (base) did not outperform ArcFace on CHIYA, indicating the need for further adaptation.  ￼

⸻

5. Proposed Fine‐Tuning Methodology

5.1. Motivation
	•	Off‐the‐shelf models are trained on “in‐the‐wild” adult faces and do not account for large age‐induced changes in adolescents.
	•	DocFace+ base model’s training did not include adolescent ID‐to‐selfie pairs, so its dynamic‐weight‐imprinting (DWI) approach was insufficient for CHIYA’s distribution.

5.2. Fine‐Tuning Strategy
	1.	Select Best Open‐Source Baseline
	•	ArcFace [7] was chosen as the starting point because it achieved the highest baseline TAR on CHIYA among open models.  ￼
	2.	Triplet‐Loss for Few‐Shot Verification
	•	Since each subject in CHIYA has only two images (ID vs. selfie), traditional classification losses (softmax) are not suitable. Instead, the authors use semi‐hard triplet loss as in FaceNet [19]:
	•	Anchor: randomly pick either an ID or selfie image.
	•	Positive: the corresponding image of the other type (ID if anchor is selfie; selfie if anchor is ID).
	•	Negative: an ID‐card image of a different subject (if anchor is ID) or a selfie of a different subject (if anchor is selfie).
	•	This enforces the network to pull ID/selfie embeddings of the same person closer and push embeddings of different people apart, directly optimizing for verification without requiring more than two samples per class.  ￼
	3.	Heterogeneous Triplet Selection
	•	All triplets are constrained so that the anchor and positive belong to opposite domains (ID vs. selfie). This ensures the network explicitly learns cross‐domain separation.  ￼
	4.	Training Regimen
	•	Use more than 230,000 subjects (≈460,000 images) for fine‐tuning.
	•	Images are resized to match ArcFace’s 112 × 112 input; alignments remain as in preprocessing.
	•	Semi‐hard triplet mining (following [19]) helps convergence despite only two images per subject.
	•	The last fully connected layer’s weights remain fixed for classes not present in the current batch (unlike DWI used by DocFace+).
	•	Learning rates and batch sizes follow standard ArcFace fine‐tuning schedules (not explicitly detailed in this summary).  ￼

⸻

6. Experimental Setup
	•	Training Partition: 230,000 subjects for fine‐tuning; remaining subjects reserved for testing across the five age‐gap subsets.
	•	Testing Protocol: For each age gap (e.g., i10s1819), compute TAR at FAR = 0.01% and FAR = 0.1% for all genuine vs. impostor pairs.
	•	Comparison Points:
	1.	Off‐the‐shelf ArcFace and other baselines (no fine‐tuning).
	2.	DocFace+ (base) without further adaptation.
	3.	DocFace+ fine‐tuned (as reported in [21]), for reference.
	4.	Proposed ArcFace‐triplet fine‐tuned on CHIYA.  ￼

⸻

7. Results and Analysis

7.1. Baseline Performance (No Fine‐Tuning)
	•	ArcFace (pretrained):
	•	Average TAR across all age gaps ≈ 85.13% at FAR = 0.01%.
	•	Performance drops dramatically for i10s1819 (≈ 15.77% TAR) and improves as age gap narrows (i18s1819 ≈ 96–97% TAR).  ￼
	•	DocFace+ (base):
	•	Underperforms ArcFace on CHIYA, indicating that DWI and sibling‐network structure trained on adult ID‐selfie data do not generalize well to adolescent ID images.  ￼
	•	COTS‐1, GOTS‐1, GOTS‐2:
	•	COTS‐1: 62.70% average TAR (FAR = 0.01%)
	•	GOTS‐1: 42.82%
	•	GOTS‐2: 58.91%
	•	Highlighting that many proprietary matchers tuned for adult faces fail under large age variation in adolescence.  ￼

7.2. DocFace+ Fine‐Tuned (Reference from [21])
	•	On CHIYA, DocFace+ fine‐tuned (separate from this paper’s method) yields improvements but still trails behind the proposed ArcFace‐triplet approach on the most difficult age gaps.  ￼

7.3. Proposed ArcFace‐Triplet Finetuning
	•	i10s1819 (ID at age 10 vs. selfie at age 18–19):
	•	Before Fine‐Tuning: TAR ≈ 15.77% (FAR = 0.01%).
	•	After Fine‐Tuning: TAR ≈ 96.67% (FAR = 0.01%)—an absolute gain of ~80.9%.  ￼
	•	i12s1819:
	•	Pre‐fine‐tune TAR ≈ 36.21%; post‐fine‐tune ≈ 97.05% (FAR = 0.01%).
	•	i14s1819:
	•	Pre‐fine‐tune TAR ≈ 63.04%; post‐fine‐tune ≈ 98.24%.
	•	i16s1819:
	•	Pre‐fine‐tune TAR ≈ 82.07%; post‐fine‐tune ≈ 99.12%.
	•	i18s1819:
	•	Pre‐fine‐tune TAR ≈ 95.13%; post‐fine‐tune ≈ 99.74%.  ￼
	•	Average Gains: Across all age‐gap subsets, the fine‐tuned ArcFace model improves TAR by approximately 30–80 percentage points (depending on the subset) at the same FAR (0.01%).  ￼
	•	Cross‐Dataset Generalization: When evaluated on the Public‐IvS dataset (a public subset of CASIA‐IvS simulated ID‐selfie data), the fine‐tuned model maintains competitive performance, surpassing DocFace+ and other baselines.  ￼

7.4. Gender Analysis
	•	No consistent or large gender bias in verification accuracy was observed. Both male and female adolescent pairs achieved similar TAR after fine‐tuning, suggesting that the triplet‐loss fine‐tuning on CHIYA mitigates gender‐based performance gaps.  ￼

⸻

8. Key Contributions
	1.	First Study on Adolescent ID‐to‐Selfie Matching
	•	Demonstrates that face matchers’ accuracy degrades significantly when the ID image is from early adolescence, motivating the need for specialized training.  ￼
	2.	Large Private Dataset (CHIYA)
	•	Release of a 263,490‐subject dataset capturing ID vs. selfie pairs across ages 9–19. This is, to the authors’ knowledge, the largest such dataset focusing on adolescence.  ￼
	3.	Triplet‐Loss Fine‐Tuning Strategy
	•	Introduces a simple yet effective method: taking an off‐the‐shelf ArcFace model and fine‐tuning it using a heterogeneous triplet sampling strategy to directly address cross‐domain (ID vs. selfie) and age‐gap issues.
	•	Achieves dramatic improvements (e.g., 15.77% → 96.67% TAR on age 10 vs. 18–19 subset at FAR = 0.01%).  ￼
	4.	Publicly Available Model
	•	The fine‐tuned ArcFace model (and training scripts) have been released on GitHub, enabling other researchers to apply this approach or benchmark against it.  ￼ ￼
	5.	Gender‐Neutral Performance
	•	Finds no large gender‐based accuracy differences among adolescents, indicating that the model generalizes equally well to male and female subjects.  ￼

⸻

9. How the Method Works (Step by Step)
	1.	Start with Pretrained ArcFace
	•	Use a publicly released ArcFace model (e.g., ResNet100 backbone) pretrained on large “in‐the‐wild” adult face datasets.  ￼
	2.	Prepare Training Triplets
	•	For each of ≈ 230,000 subjects in CHIYA:
	•	Anchor: randomly pick either the ID‐card face or the selfie face.
	•	Positive: the other image (ID if anchor was selfie, selfie if anchor was ID).
	•	Negative: randomly sample an ID (if anchor is ID) or a selfie (if anchor is selfie) from a different subject.
	•	Only include triplets where anchor and positive come from opposite domains (enforcing cross‐domain learning).  ￼
	3.	Fine‐Tune Network with Triplet Loss
	•	Feed triplets into ArcFace’s feature extraction network.
	•	Compute embeddings (512‐dim vectors) for anchor, positive, and negative.
	•	Apply semi‐hard triplet mining: choose negatives that are harder (i.e., their distance to the anchor is smaller than that of the positive, but still > margin).
	•	Use triplet loss:
\mathcal{L} = \max\bigl(\lVert f(a) - f(p)\rVert_2 \;-\; \lVert f(a) - f(n)\rVert_2 \;+\;\alpha,\;0\bigr),
where f(\cdot) is the embedding function and \alpha is a margin hyperparameter.  ￼
	4.	Iterate Until Convergence
	•	Use standard Adam/SGD optimizers with a learning rate schedule (e.g., start at 1e-4, decay by 0.1 every few epochs).
	•	Monitor validation TAR on a hold‐out subset of CHIYA; stop when improvements plateau.  ￼
	5.	Inference (Deployment)
	•	For any new pair (ID vs. selfie), detect and align faces (RetinaFace), resize to 112 × 112, extract embeddings via fine‐tuned ArcFace, then compute cosine similarity.
	•	If similarity > threshold (chosen to yield FAR = 0.01%), accept as match; else reject.  ￼

⸻

10. Conclusions
	•	Significant Accuracy Gains: Fine‐tuning ArcFace with a heterogeneous triplet strategy on CHIYA increased TAR from as low as ≈ 15.77% to ≈ 96.67% (i10s1819) at FAR = 0.01%. Across all subsets, post‐fine‐tune TAR > 96%.
	•	Practical Impact: This methodology enables robust real‐world applications—such as government benefit enrollment—where verifying adolescent identities via legacy ID photos is critical.
	•	Generalization: The model also performs strongly on other ID‐Selfie benchmarks (e.g., Public‐IvS), indicating wide applicability.
	•	Open Resources: By releasing the fine‐tuned model and code, the authors encourage further research in adolescent face matching, a previously underexplored domain.  ￼ ￼

References:
	1.	V. Albiero, N. Srinivas, E. Villalobos, J. Perez‐Facuse, R. Rosenthal, D. Mery, K. Ricanek, K. W. Bowyer, Identity Document to Selfie Face Matching Across Adolescence, arXiv:1912.10021 (2020).  ￼ ￼
	2.	Y. Shi & A. K. Jain, DocFace+: ID Document to Selfie Matching, IEEE Trans. Biometrics, Behavior, and Identity Science (2019).  ￼ ￼
	3.	Q. Deng et al., ArcFace: Additive Angular Margin Loss for Deep Face Recognition, 2019.  ￼
