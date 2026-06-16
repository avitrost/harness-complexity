# Patch/editing affordance mechanism

Status: inconclusive. Current codex_full_no_patch_tool treatment is contaminated because exec_command still intercepts shell apply_patch commands and applies them via harbor_adapter.

Required follow-up:
1. Add true codex_full_no_patch_affordance profile.
2. Set patch_tool=False.
3. Remove/condition apply_patch prompt instructions when patch tool is unavailable.
4. Disable exec_command apply_patch interception for that profile, or return hard failure with blocked_apply_patch_cmd=true.
5. Run same TBLite 100 x1 gpt-5.4-mini low paired against codex_full in Slurm.

Decision rule: patch causality supported only if true no-patch drops and paired losing traces show forced shell-edit fallback causing edit failures/corruption.
