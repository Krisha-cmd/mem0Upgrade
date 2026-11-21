"""PrefEval-faithful judge + false-application judge. Implemented in Phase 4.

Verified PrefEval protocol (generation_task/llm_based_evaluation_errortypes.py and
get_preference_following_accuracy_generation_task.py @ 50795054), to be reproduced exactly:
- four prompts (prompts/judge_prefeval/): acknow, violate, hallucinate (uses acknow's extracted
  preference), helpful; XML output <explanation>/<preference> + <answer>; temperature 0.
- is_ack = "yes" in acknow; is_hall = is_ack and "yes" in hallucinate; is_viol = "yes" in violate;
  is_unhelpful = "no" in helpful
- inconsistent = ack & !hall & viol & !unhelpful; hallucination_of_pref_violation = ack & hall &
  viol & !unhelpful; preference_unaware_violation = !ack & viol & !unhelpful
- followed = not any(inconsistent, hall_violation, unaware_violation, unhelpful)
"""

from __future__ import annotations


def main(argv=None) -> None:
    raise SystemExit("judge stage is implemented in Phase 4")


if __name__ == "__main__":
    main()
