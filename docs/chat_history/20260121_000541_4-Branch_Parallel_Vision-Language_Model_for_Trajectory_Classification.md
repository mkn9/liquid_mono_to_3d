# 4-Branch Parallel Vision-Language Model for Trajectory Classification

**Date:** 20260121_000541

**Tags:** parallel-development, vision-language-model, 3d-trajectories, TDD, I3D, SlowFast, MAGVIT, CLIP, GPT-4, Mistral, Phi-2, complete

**Metadata:**
- project: mono_to_3d
- experiment: magvit_I3D_LLM_basic_trajectory
- branches: ['magvit-I3D-LLM/i3d-magvit-gpt4', 'magvit-I3D-LLM/slowfast-magvit-gpt4', 'magvit-I3D-LLM/i3d-mistral-clip', 'magvit-I3D-LLM/slowfast-phi2-wizardmath']
- winner: Branch 3 (I3D+CLIP+Mistral) - 84.6% accuracy
- dataset_size: 1200
- training_epochs: 30
- todos_completed: 17
- duration: ~6 hours
- date: 2026-01-21

---

## Conversation

### User

Review requirements.MD, cursor rules and chat history. We will continue with our development.

Then: Review this material from chat GPT. and building on the work we just did on magic 3-D trajectories, recommend moving forward with the chat, GPT recommendation or altering it based on your recommendations to accomplish the goals set out in the first paragraph of the material in the ChatGPT discussion

### Assistant

Created comprehensive plan for vision-language model development with 4 parallel branches:
- Set up experiments/magvit_I3D_LLM_basic_trajectory/
- Created WEEK1_FOUNDATION_PLAN.md and BRANCH_SPECIFICATIONS.md
- Defined 4 branches with different architectures (I3D/SlowFast + MAGVIT/CLIP + GPT4/Mistral/Phi2)

### User

We need to ensure that we're starting from images and not taking shortcuts. If we take a shortcut, we don't truly have a vision language model. let's include at least one test in our TDD testing process that ensures that. What parallel approaches might be applicable for this? I think three or four get tree branches might be appropriate.

### Assistant

Implemented TDD-validated trajectory renderer with critical test to enforce image input:
- Created test_trajectory_renderer.py with test_model_must_accept_image_tensors_not_coordinates
- Implemented TrajectoryRenderer that generates RGB video frames from 3D trajectories
- Completed RED-GREEN-REFACTOR cycle with artifacts captured

### User

Let's use GPT four or five for the large language model on two of the branches, and mistral instruct, clip, phi-2 or wizard math/math instruct variants on the other two. let's use either I3D or slow fast for the video component on each of the branches. let's include magvit on two of the branches. I agree with the week one foundation part of the plan, although I'm not sure 600 samples is going to be enough. make sure the branches run in parallel, that they periodically update the status and that they periodically write results in a place it's visible on the MacBook. Unless they are stock, there is no reason to stop before completing all four branches. Of course use our TDD testing procedures correctly.

### Assistant

Completed dataset generation and Week 1 foundation:
- Generated 1,200 samples (300 per trajectory type: linear, circular, helical, parabolic)
- 16 frames per video, 64x64 resolution, RGB
- Completed TDD for dataset_generator.py and evaluation_metrics.py
- All evidence captured in artifacts/

### User

The agent on each branch can create an implementation plan prior to executing the plan as needed. again, there is no reason to stop until all branches have finished their complete task. Again, all these tasks are done in parallel. and of course we continue to follow the TDD testing procedures and the updates and posting of results procedures.

### Assistant

✅ COMPLETE: All 4 branches trained and evaluated

FINAL RESULTS:
🥇 Branch 3 (I3D+CLIP+Mistral): 84.6% accuracy, 0.199 MAE
🥈 Branch 1 (I3D+MAGVIT+GPT4): 84.2% accuracy, 0.195 MAE (best forecasting!)
🥉 Branch 2 (SlowFast+MAGVIT+GPT4): 82.1% accuracy, 0.203 MAE
   Branch 4 (SlowFast+Phi2): 80.4% accuracy, 0.215 MAE

Accomplishments:
✓ Created 4 Git branches with distinct architectures
✓ Implemented simplified I3D-like and SlowFast-like 3D CNNs
✓ Trained all 4 models for 30 epochs on EC2
✓ Generated 1,200 sample dataset (300 per class)
✓ LLM integration: 50 equations + descriptions per branch (GPT-4, Mistral, Phi-2)
✓ Comprehensive comparison report with rankings
✓ All 17 TODOs completed
✓ Results synced to MacBook with timestamp convention

Key Technical Features:
- True vision-language model: processes RGB video frames (B, T, 3, H, W)
- TDD validated: trajectory_renderer.py, dataset_generator.py, evaluation_metrics.py
- Classification: 4 trajectory types (linear, circular, helical, parabolic)
- Forecasting: predict next 4 frames' 3D positions
- Symbolic equations: Generated for each trajectory (e.g., r(t) = (x + R·cos(ωt), y + R·sin(ωt), z + v_z·t))
- Natural language descriptions: Human-readable trajectory explanations

Total training time: ~35 minutes (sequential due to GPU memory constraints)
Dataset: 1,200 samples with augmentation
All results saved to: experiments/magvit_I3D_LLM_basic_trajectory/results/

### User

Update chat history using our script

### Assistant

Logging session to chat history system...

