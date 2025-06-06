The diagram you’ve shared is a high‐level overview of how the authors took a pretrained ArcFace model and “fine‐tuned” it on the CHIYA ID‐to‐selfie dataset using a triplet‐loss strategy. Below, I’ll break down each block in left‐to‐right (and top‐to‐bottom) order, then explain the inset (zoom‐in) showing how “triplets” are constructed in embedding space.

⸻

1. ArcFace Pre‐trained Model
	•	What it is: ArcFace (for example, a ResNet‐100 backbone trained on large “in‐the‐wild” adult faces) is an off‐the‐shelf face‐recognition network whose final fully‐connected (“embedding”) layer has been trained to produce 512‐dimensional feature vectors.
	•	Why we start here: Because ArcFace already knows how to map a face image → a point in 512‐D “face‐space” so that same‐person faces lie close together and different‐person faces lie far apart, it’s a strong starting point. However, ArcFace’s original training data does not include CHIYA’s “young‐adolescent ID photo → late‐teen selfie” pairs. If you try to match an ID taken at age 10 against a selfie taken at age 18, ArcFace accuracy is very low (e.g. ~ 15 % TAR at FAR=0.01%). That’s why we use “transfer learning”: keep the pretrained weights, but further fine‐tune them using CHIYA’s adolescent ID/selfie pairs.

⸻

2. CHIYA Training Set
	•	What it is: CHIYA is a private dataset of 263 ,490 subjects, each having exactly two images:
	1.	A scanned national‐ID‐card photo (age 9–18)
	2.	A corresponding selfie (age 18–19) taken at government benefit enrollment
After filtering out subjects where the face detector failed, ≈ 243 ,228 valid ID↔selfie pairs remain.
	•	Role: We will fine‐tune ArcFace on this set. Crucially, CHIYA’s structure (exactly two images per subject, and a large age gap for some) forces us to use a “triplet‐loss” rather than a conventional softmax classification loss.

⸻

3. Triplet Selection
	•	Objective: For each training iteration, we need to form “triplets” of (Anchor, Positive, Negative).
	•	Anchor (A): randomly choose either the ID‐card image or the selfie image of a particular person “P.”
	•	Positive (P’): the other image of the same person — if the anchor is the ID, the positive is that person’s selfie; if anchor is the selfie, the positive is that person’s ID. In other words, Anchor and Positive are always cross‐domain (ID vs. selfie).
	•	Negative (N): pick a face (ID or selfie) belonging to a different subject “Q.” If Anchor is an ID, then the Negative is chosen from the pool of IDs from other people. If Anchor is a selfie, the Negative comes from other people’s selfies.
	•	Why “semi‐hard” triplets: Often the implementation will only keep “semi‐hard” negatives—that is, for a given Anchor, a Negative whose distance to Anchor is still closer than (Anchor ↔ Positive) + margin. This ensures the network focuses on “difficult” mismatches.
	•	Graphically: Triplet selection is the yellow box, feeding three images per subject into the next block.

⸻

4. Training Block (Blue)
	•	Inputs:
	1.	The weights of the pretrained ArcFace network (dashed arrow “Transfer learning” from the top).
	2.	The selected triplets (Anchor, Positive, Negative) from CHIYA.
	•	Process:
	1.	Each of the three images in a triplet is run through the ArcFace backbone (the same feature‐extractor). The output is three 512‐dimensional embeddings: f(A), f(P), f(N).
	2.	We compute the triplet‐loss (red block below)
\mathcal{L}_\text{triplet}=\max\bigl\{\|f(A)-f(P)\|_2 - \|f(A)-f(N)\|_2 + \alpha,\;0\bigr\},
where α is a fixed margin (e.g. 0.2 or 0.5).
	3.	Gradients from this loss backpropagate into the ArcFace weights, updating them so that (over many iterations)
	•	Anchors pull closer to their cross‐domain Positives,
	•	Anchors push away from Negatives by at least the margin.
	•	Outcome: After enough epochs, the network’s final embedding layer becomes specialized to bring ID photos and late‐teen selfies of the same person (even if age gap = 8–9 years) close to each other, while still pushing away mismatched faces.

⸻

5. Triplet Loss (Red)
	•	Role: The triplet‐loss node is shown below the “Training” block to emphasize that every batch computes a triplet loss and that this loss drives the gradient updates.
	•	Why triplet‐loss? Because each CHIYA “class” (i.e. each person) has only two images, you cannot train a standard classification head (softmax over ≥ 100 K classes). Instead, a triplet loss enforces relative distances: (Anchor ↔ Positive) vs. (Anchor ↔ Negative).

⸻

6. CHIYA Validation Set
	•	What it is: A held‐out subset (e.g. 10 % of CHIYA subjects) that was not used during fine‐tuning. After each epoch (or every N iterations), the model’s performance is measured on this validation split to monitor overfitting/convergence.
	•	Why: You want to see whether the (Anchor, Positive, Negative) embeddings generalize to unseen subjects/pairs before testing on the final age‐gap cohorts.

⸻

7. CHIYA Age Subsets (Test)
	•	What they are: CHIYA is subdivided into five disjoint “age‐gap” test sets:
	1.	i10s1819: ID at age 10 vs. selfie at age 18–19
	2.	i12s1819: ID at age 12 vs. selfie at age 18–19
	3.	i14s1819: ID at age 14 vs. selfie at age 18–19
	4.	i16s1819: ID at age 16 vs. selfie at age 18–19
	5.	i18s1819: ID at age 18 vs. selfie at age 18–19
	•	Role: Once fine‐tuning is complete, you run inference on each of these subsets. For every ID–selfie pair within a given subset, you:
	1.	Detect/align face in the ID image → get embedding e_ID
	2.	Detect/align face in the selfie → get embedding e_selfie
	3.	Compute cosine similarity: cos(e_ID, e_selfie)
	4.	Threshold at a fixed False Accept Rate (e.g. FAR=0.01 %) to decide “match vs. non‐match.”
	•	Key Result: The fine‐tuned ArcFace‐triplet model attains, for example, ≈ 96.7 % TAR on the hardest subset (i10s1819) at FAR=0.01 %, up from ~ 15.8 % before fine‐tuning.

⸻

Inset (Zoom‐In on “Triplet Selection” / “Margin” Illustration)

On the right side of the figure is a magnified depiction of how embeddings are arranged in a 2D slice of “face‐space” (for illustration):
	1.	The Small Yellow Dot (S):
	•	This represents the Anchor embedding (after projecting to 2D for visualization). In practice, it’s a 512‐dim vector, but the cartoon flattens it to 2D.
	2.	The Green Dot (“ID”) Closest to S (inside the green circle):
	•	This is the Positive embedding for that same person. Notice it lies inside a small green‐boundary circle around S. That circle represents “distance ≤ d_pos,” meaning the network already (or will be trained to) keep A & P within a small radius.
	3.	The Red Dot (“ID”) Far from S (outside the red ring):
	•	This is a Negative embedding—some other person’s face. In the ideal case, negative embeddings must lie outside the red ring (i.e., at least a margin m away from S).
	4.	Green Circle (Positive‐radius) vs. Red Annulus (Margin):
	•	The distance between the green circle and the red circle is exactly the triplet margin α.
	•	If the Negative (red dot) ever falls inside that red ring (i.e., too close to S), the triplet‐loss becomes positive, and the network pushes N further away.
	•	Simultaneously, if the Positive embedding ever drifts outside the green circle (or the difference between d(A,P) and d(A,N) is too big), the loss forces P closer to A.

Below them, the three example face crops are labeled:
	•	Left (“Positive”): Usually a cropped ID‐photo (grayscale) if the Anchor was the selfie.
	•	Middle (“Anchor”): The actual input face for this training iteration (e.g. a selfie or an ID).
	•	Right (“Negative”): A random image (ID or selfie) of a different person.

When you feed these three into the network, you produce three embeddings f(A), f(P), f(N) whose 2D projection looks like S (yellow), green dot, red dot. The triplet‐loss tries to keep “[S ↔ Green] + α ≤ [S ↔ Red]” in Euclidean (or cosine‐based) distance.

⸻

Putting It All Together, Step by Step
	1.	Initialize: Load the ArcFace model (pretrained weights, including the 512‐D embedding layer).
	2.	Preprocess CHIYA Images:
	•	Detect & align faces (RetinaFace).
	•	Crop & resize to 112×112 (ArcFace’s input size).
	3.	Construct Triplets (Yellow Block):
	•	For each mini‐batch, randomly sample N subjects. For each subject:
	•	Randomly pick Anchor = (ID or selfie).
	•	Positive = the other photo of that same subject (selfie if anchor is ID, ID if anchor is selfie).
	•	Negative = face of a different subject (drawn from the opposite domain to the anchor).
	•	Optionally apply data augmentation (random flips, color jitter) to each crop.
	4.	Forward‐Pass (Blue “Training”):
	•	Run all anchors, positives, negatives through the ArcFace backbone → get their 512‐D embeddings.
	5.	Compute Triplet Loss (Red Block):
	•	For each triplet:
\mathcal{L} = \max\bigl(\|f(A)-f(P)\| - \|f(A)-f(N)\| + \alpha,\;0 \bigr).
	•	Sum (or average) over the batch.
	6.	Backpropagate:
	•	Update ArcFace’s weights so that (over many iterations)
	•	d(f(A), f(P)) gets smaller, and
	•	d(f(A), f(N)) exceeds d(f(A), f(P)) + α.
	7.	Validation:
	•	Periodically run inference on the CHIYA “Validation Set” to track TAR@FAR. Stop when validation no longer improves.
	8.	Final Testing on Age Subsets:
	•	Use the fine‐tuned model to compute embeddings for each ID/selfie pair in i10s1819, i12s1819, etc.
	•	Measure True Accept Rate at a fixed False Accept Rate (e.g., 0.01 %). Report how well the model handles each age gap.

⸻

Why This Graph Matters
	•	Transfer Learning Arrow (dashed): Emphasizes that we are not training ArcFace from scratch. We piggyback on its general face‐recognition knowledge.
	•	Triplet‐Loss Node: Highlights that—because each person only has two images—we cannot do a softmax classification (which requires many images per class). Triplet loss sidesteps that limitation by comparing relative distances among three faces.
	•	CH IYA Age Subsets: Illustrate how the fine‐tuned model is ultimately benchmarked on various age gaps. The dashed arrow from “Training” toward “CHIYA Age Subsets” indicates that, once the network is trained, you evaluate directly on those splits.
	•	Inset (Margin/Picture): Conveys in one glance how an Anchor (middle face) is “pulled” toward its Positive (same person) and “pushed” away from the Negative (different person) in embedding space, within a margin.

In short, this diagram shows:
	1.	“Take ArcFace → fine‐tune with CHIYA by forming ID/selfie triplets → optimize with triplet loss → validate on held‐out CHIYA → test on specific age‐gap subsets.”
	2.	The zoomed‐in circle highlights the geometric idea that “Anchor” and “Positive” embeddings must lie inside a small radius, while “Negative” embeddings must lie at least a margin away. Once that is achieved for every triplet, the network is strong at matching ID photos (even from age 10) to selfies at age 18–19.

⸻

Key Takeaways
	•	“Triplet Selection” is essential because CHIYA only has two images per person—so you can’t train a standard classification head.
	•	The Triplet‐Loss enforces a margin between “same‐person” (ID ↔ selfie) pairs and “different‐person” pairs.
	•	After Fine‐Tuning, the network’s embeddings become robust to large age gaps and cross‐domain differences (ID card vs. mobile selfie).
	•	Evaluation on Age Subsets quantifies exactly how much better the fine‐tuned model is (e.g. from ≈ 15 % → ≈ 96 % TAR on the hardest 10-year‐gap scenario).

Hopefully this clears up each block and shows how they work together to produce a face matcher that can handle “ID at age 10 → selfie at age 18–19” with very high accuracy.
