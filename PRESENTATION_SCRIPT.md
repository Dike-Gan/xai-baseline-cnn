# Presentation Script - Slide by Slide

## SLIDE: Presentation Outline

### 🎤 What to Say:

---

**[Opening - 10 seconds]**

"Good morning/afternoon everyone. Today I'll walk you through our research on explainable AI for image classification, which is divided into two main parts."

---

**[Section 1 - 15 seconds]**

"**First, Baseline Model Construction.** We start by asking: *How do different pretraining sources and training strategies affect CNN performance?*

To answer this, we designed a systematic experiment comparing four ResNet-18 models under controlled factors:
- Two pretraining sources: ImageNet and Flowers-102
- Two training strategies: Linear Probing and Fine-Tuning

This 2-by-2 experimental design allows us to isolate the effect of each factor and their interaction.

We'll examine the performance results and discuss the limitations of these baseline models."

---

**[Section 2 - 15 seconds]**

"**Second, Explainability-Driven Analysis.** After establishing our baseline, we move beyond just accuracy numbers.

We perform difficulty-based sample categorization to identify where models struggle.

We analyze high-confidence wrong predictions - cases where the model is confidently incorrect, which are particularly problematic in real applications.

And we use visual explanations like Grad-CAM to understand what features the model is actually looking at, revealing feature-level patterns that explain both successes and failures."

---

**[Transition - 5 seconds]**

"This two-stage approach gives us both quantitative performance metrics AND qualitative insights into model behavior.

Let's begin with the baseline model construction."

---

### ⏱️ **Total Time: ~45 seconds**

---

## 💡 Key Points to Emphasize:

1. **Systematic Design**: "controlled factors" - shows scientific rigor
2. **Research Question**: Clearly state what you're investigating
3. **Two-Stage Approach**: Performance first, then explainability
4. **Practical Relevance**: Mention "high-confidence wrong predictions" - real-world concern

---

## 🎯 Body Language & Delivery Tips:

- **Make eye contact** when saying "two main parts" (hold up 2 fingers)
- **Gesture to outline** as you go through each section
- **Emphasize numbers**: "Four models," "2-by-2 design"
- **Pause** between Section 1 and Section 2 (visual transition)
- **Confident tone** on "controlled factors" and "systematic experiment"

---

## 🔑 Backup Points (If Asked Questions):

**Q: "Why these specific factors?"**
A: "Pretraining source and training strategy are two of the most critical design choices in transfer learning. By systematically varying both, we can understand their individual effects and whether they interact."

**Q: "Why two stages?"**
A: "Stage 1 establishes baseline performance - which model is best. Stage 2 explains WHY - what features lead to correct vs incorrect predictions. This combination of performance and explanation is essential for trustworthy AI."

**Q: "What's the practical application?"**
A: "Understanding when and why models fail helps us improve them. High-confidence errors are particularly dangerous in applications like medical diagnosis or autonomous driving, so identifying these patterns is crucial."

---

## 📝 Alternative Version (Shorter - 30 seconds):

"Today I'll present our research in two parts.

**First:** Baseline Model Construction - we compare four CNN models with different pretraining sources and training strategies to understand what affects performance.

**Second:** Explainability Analysis - we go beyond accuracy to understand WHY models succeed or fail, using difficulty categorization and visual explanations.

This approach gives us both performance metrics and actionable insights. Let's begin."

---

## 📝 Alternative Version (More Technical - 60 seconds):

"Today's presentation covers two complementary research stages.

**Stage 1: Baseline Model Construction** addresses the research question: *How do different pretraining sources and training strategies affect CNN performance on household object classification?*

We employ a 2-by-2 factorial design with ResNet-18 architecture:
- **Factor 1**: Pretraining source - ImageNet-1K versus Flowers-102
- **Factor 2**: Training strategy - Linear Probing versus Fine-Tuning

This yields four models that allow us to quantify main effects and interaction effects.

**Stage 2: Explainability-Driven Analysis** applies XAI methods to understand model behavior:
- Difficulty-based categorization stratifies samples by confidence
- Error analysis identifies systematic failure modes
- Grad-CAM visualizations reveal spatial attention patterns

Together, these stages provide both quantitative benchmarking and qualitative interpretation.

Let's examine the baseline construction methodology."

---

## 🎨 Visual Aids to Use:

While saying this, you could:
1. **Point to "1. Baseline Model Construction"** when discussing Stage 1
2. **Point to "2. Explainability-Driven Analysis"** when discussing Stage 2
3. **Use hand gestures** for "four models" (4 fingers) and "2-by-2" (draw 2x2 grid in air)
4. **Circle "controlled factors"** with laser pointer if available

---

## ✅ Checklist Before This Slide:

- [ ] Make eye contact with audience
- [ ] Speak clearly and at moderate pace
- [ ] Don't read verbatim from slide
- [ ] Use confident, professional tone
- [ ] Transition smoothly to next slide

---

## 🔄 Common Mistakes to Avoid:

❌ **DON'T** just read the bullet points word-for-word
❌ **DON'T** rush through this slide - it sets up your entire talk
❌ **DON'T** use jargon without explanation (define "linear probing" later)
❌ **DON'T** apologize or say "I'm nervous"

✅ **DO** speak naturally and conversationally
✅ **DO** emphasize the research question
✅ **DO** show enthusiasm about the systematic design
✅ **DO** preview the value of the explainability stage

---

## 🎯 Key Takeaway for Audience:

After this slide, the audience should understand:
1. **Two-stage structure**: Performance → Explanation
2. **Systematic approach**: Controlled experiment with 4 models
3. **Practical value**: Not just "which is best" but "why it works"

---

Good luck! 🎉

