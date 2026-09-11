# 50-Interview Project Plan

## 1. Purpose

Turn the existing resume-aware voice interviewer into a reliable system that can conduct **50 complete, useful technical interviews**. A useful interview means that it remembers the candidate throughout the session, asks role-appropriate non-repetitive questions, follows up on the candidate's actual answer, and produces an evidence-based result.

This is an improvement plan, not a rewrite. The current FastAPI UI, speech-to-text relay, resume extraction, RAG engine, interview FSM, LLM question phrasing, and TTS remain the foundation.

## 2. Definition of Success

For a completed interview, the system must:

1. Create one interview session and use it for every candidate turn.
2. Extract a candidate profile from the resume and use only supported skills, projects, and domains for personalisation.
3. Ask one clear question at a time.
4. Ask 12 to 15 questions in a 20 to 30 minute interview, unless the candidate ends it earlier.
5. Adapt after strong, weak, skipped, or repeat-request answers.
6. Avoid asking the same question or a near duplicate in the same session.
7. Save the transcript, questions, answers, decisions, and final evaluation.
8. Produce a final report with strengths, gaps, topic scores, and recommended next steps.

The first milestone is 50 completed pilot interviews across the selected role tracks. This is not necessarily 50 concurrent users; concurrency is a separate load-testing goal.

## 3. Current System and What Must Change

### Keep

- FastAPI server and static browser UI.
- Nemotron streaming STT and transcript correction.
- Resume parsing, hotwords, and TF-IDF retrieval.
- Interview FSM, intent detection, question engine, Qwen question generation, and TTS.

### Fix before pilot interviews

- **Session continuity:** The UI must call `POST /api/start-interview`, retain the returned `session_id`, and include it in every `/api/interview-turn-stream` request. Currently it does not, so every turn becomes a new interview.
- **One canonical interview path:** Keep the streamed SSE endpoint as the live path. Refactor or retire the older non-streaming endpoint so state is updated in only one place.
- **Prompt settings:** Either inject `system_prompt.txt` into the active prompt builder or remove the settings UI until it changes live behavior.
- **Persistence:** Replace in-memory sessions with a database-backed interview record so a restart does not lose an active or completed interview.
- **Configuration:** Move endpoint URLs, model names, CORS origins, and secrets to environment variables. Add `requirements.txt` and a startup README.

## 4. Interview Design

### Supported tracks for the pilot

Start with four tracks already reflected in the product UI and question bank:

- AI / ML Engineer
- Backend Engineer
- Full-Stack Engineer
- Frontend Engineer

Each track needs an explicit competency map. A question must test one competency, at one difficulty level, with an expected answer shape and scoring rubric.

### Interview stages

| Stage | Questions | Goal |
| --- | ---: | --- |
| Introduction and resume verification | 1 | Confirm the candidate's project ownership and target role. |
| Project deep dive | 3 | Test real implementation choices, constraints, and individual contribution. |
| Core technical skills | 4 to 5 | Test the role's required concepts from fundamentals to trade-offs. |
| Problem solving or debugging | 2 to 3 | Test structured reasoning, edge cases, and recovery from uncertainty. |
| Behavioral and closing | 2 | Test collaboration, learning, and candidate questions. |

The system should not mechanically force every stage. A junior candidate may need more fundamental questions; an experienced candidate may move quickly into architecture and trade-offs.

### Question quality contract

Every generated or retrieved question must meet these rules:

1. It is grounded in the resume, selected role, previous answer, or an explicit role requirement.
2. It has one primary intent. Do not combine two unrelated questions.
3. It is answerable aloud in roughly one to two minutes.
4. It requests evidence: a decision, implementation detail, metric, trade-off, failure mode, or result.
5. It does not invent a project, tool, employer, or accomplishment.
6. It has a defined expected-concept list and rubric before it is asked.
7. A follow-up deepens the last answer; it does not jump topics without a reason.
8. A candidate's uncertainty leads to a supportive pivot, not a penalty-only loop.

### Question patterns to prefer

- “You mentioned _X_. What problem did it solve, and what trade-off did you make?”
- “Walk me through the component you personally owned. How did data move through it?”
- “What would fail first at ten times the current usage, and how would you measure it?”
- “How did you know the model or feature was improving? Which metric mattered and why?”
- “Describe a production or project failure. How did you diagnose, fix, and prevent it?”

Avoid generic filler such as “Tell me more about Python” or unsupported assumptions such as “How did you deploy Kubernetes?” when Kubernetes is not in the resume.

## 5. Question Bank Plan

The existing bank is a good seed, but 83 broad questions are not enough for consistently strong 50-interview coverage. Build a curated bank of **400 to 600 questions** before the full pilot.

Each question record should include:

```json
{
  "id": "ml-model-evaluation-03",
  "track": "aiml",
  "competency": "model_evaluation",
  "topic": "classification_metrics",
  "difficulty": 2,
  "question": "For your classifier, why did you choose F1 over accuracy, and how did that affect the threshold you used?",
  "expected_concepts": ["class imbalance", "precision", "recall", "threshold"],
  "follow_up_ids": ["ml-model-evaluation-04"],
  "anti_patterns": ["claims metric without explaining why"],
  "source": "curated"
}
```

Recommended initial coverage:

- 100 AI / ML questions: data, modelling, evaluation, deployment, LLM/RAG, debugging.
- 100 Backend questions: API design, databases, concurrency, reliability, security, observability.
- 100 Full-Stack questions: frontend state, API integration, performance, authentication, deployment.
- 70 Frontend questions: React, browser behavior, accessibility, performance, testing.
- 50 cross-functional questions: debugging, system design, collaboration, ownership, communication.

Tag every question by track, competency, difficulty (1 to 5), expected concepts, and follow-up relationship. Use retrieval only from the candidate's eligible role track and skills. Use the LLM to phrase or personalize a chosen question, not to invent the assessment policy.

## 6. Interview Decision Policy

The backend must decide what to assess; the LLM should only produce natural spoken wording.

For each answer, the decision order is:

1. Validate the active `session_id` and load its interview state.
2. Correct the transcript only when correction confidence is high; retain the raw transcript as evidence.
3. Detect intent: repeat request, skip/unknown, self-introduction, or answer.
4. Evaluate the answer against the current question's rubric.
5. Update competence score, difficulty, topic coverage, and time/question budget once.
6. Choose one action: repeat, probe deeper, continue, pivot, change stage, or close.
7. Retrieve eligible questions, excluding questions and near duplicates already asked.
8. Generate a concise spoken form of the selected question.
9. Persist the complete turn before returning it to the browser.

Difficulty rules should be explicit:

- Score 4 or 5 out of 5: deepen the topic or increase difficulty by one.
- Score 3: move to the next competency at the same difficulty.
- Score 1 or 2: ask one clarifying/fundamental follow-up, then pivot if needed.
- Skip or “I do not know”: mark a gap, reduce difficulty, and pivot immediately.
- Repeat request: replay the exact last question without scoring the request as an answer.

## 7. Scoring and Final Report

Score answers with a transparent 1 to 5 rubric rather than relying on answer length alone.

| Dimension | What it measures |
| --- | --- |
| Correctness | Is the technical explanation accurate? |
| Depth | Does the candidate explain mechanisms, limits, and choices? |
| Evidence | Do they provide concrete implementation details, metrics, or examples? |
| Trade-offs | Do they recognise alternatives and consequences? |
| Communication | Is the answer structured and understandable? |

The final report should show:

- Overall recommendation: strong hire, hire, mixed, no hire, or insufficient evidence.
- Per-competency score and confidence level.
- Evidence-linked strengths: quote or reference the answer turn that supports each conclusion.
- Evidence-linked gaps and suggested learning topics.
- Questions asked, questions skipped, and topics not assessed.
- A clear disclaimer that this is decision support, not an automatic hiring decision.

Do not infer protected traits, personality diagnoses, or employability from speech accent, gender, name, or background.

## 8. Data Model and Privacy

Create durable records for `CandidateProfile`, `InterviewSession`, `InterviewTurn`, `Question`, and `FinalReport`.

Minimum session fields:

- session ID, created/updated timestamps, role track, state, question budget, and consent status.
- resume-derived profile with source fields clearly separated from inferred fields.
- raw and corrected transcript for every turn.
- question ID, question text, evaluation, policy action, and timing for every turn.
- final report and user feedback.

Use a local SQLite database for the pilot. Add candidate controls to delete their interview and resume data. Restrict CORS to the deployed UI origin before exposing the app beyond local testing.

## 9. Delivery Roadmap

### Phase 0 — Stabilise the existing application (1 week)

- Implement session creation, frontend storage, and session ID propagation.
- Make the streamed endpoint canonical.
- Fix duplicate state increments and duplicate utility functions.
- Add environment configuration, dependency manifest, health checks, and README.
- Add automated tests for session continuity, repeat, skip, question deduplication, and stage progression.

**Exit criterion:** A five-turn browser interview retains one session, never repeats a question unexpectedly, and shows a consistent topic/difficulty history.

### Phase 1 — Build the assessment foundation (2 weeks)

- Define four role competency maps and scoring rubrics.
- Convert the question bank to structured tagged records.
- Curate the first 250 high-quality questions and review them manually.
- Add semantic duplicate detection and track eligibility filters.
- Persist sessions and generate a first final report.

**Exit criterion:** A reviewer can inspect any question and know the role, competency, difficulty, expected evidence, and follow-up path.

### Phase 2 — Question quality and evaluator improvements (2 weeks)

- Increase the bank to 400 to 600 questions.
- Replace length-heavy evaluation with rubric-aware LLM evaluation plus deterministic guardrails.
- Add answer evidence, confidence, and “insufficient evidence” outcomes.
- Add curated test transcripts for strong, weak, skipped, vague, off-topic, and repeat answers.

**Exit criterion:** On the curated test set, the selected next action matches reviewer expectation in at least 85% of cases.

### Phase 3 — Pilot 10 interviews (1 week)

- Run with known testers across all four tracks.
- Collect rating after each interview: question relevance, clarity, fairness, repetition, and report usefulness.
- Review every bad question manually and update tags, prompts, or policies.

**Exit criterion:** Average question-relevance rating is at least 4/5 and fewer than 5% of turns contain an unwanted repeat or unsupported assumption.

### Phase 4 — Run 50 interviews and improve weekly (3 to 4 weeks)

- Conduct 50 complete pilot interviews.
- Review a statistically useful sample of transcripts each week.
- Maintain a correction log: issue, root cause, fixed rule/question/prompt, regression test added.
- Publish aggregate metrics without exposing candidate data.

**Exit criterion:** 50 completed interviews, 90% session-completion rate, 85% or higher question-relevance satisfaction, and no unresolved critical privacy or state-loss defect.

## 10. Pilot Metrics Dashboard

Track these metrics for every interview:

- Completion rate and average duration.
- Number of questions, follow-ups, pivots, repeat requests, and skips.
- Question repetition rate and unsupported-assumption rate.
- Resume grounding rate: percentage of questions linked to an actual resume field or approved role competency.
- Candidate ratings: relevance, clarity, fairness, naturalness, and usefulness of final report.
- Human-review agreement with answer scores and hiring recommendation.
- STT correction rate and correction error rate.
- LLM, STT, and TTS latency plus failure/fallback rate.

## 11. Non-Negotiable Acceptance Tests

Before the 50-interview pilot, these tests must pass:

1. A client starts an interview once and all subsequent turns update the same session.
2. Restarting the service does not erase completed interview records.
3. A repeat request returns the exact previous question and does not consume a question budget.
4. A skipped topic leads to a respectful pivot and does not ask the same question again.
5. Every question has a track, competency, difficulty, and expected concepts.
6. The app refuses or safely handles an invalid/expired session ID.
7. Final report conclusions link back to actual answers, not invented evidence.
8. Deleting a candidate record removes their stored resume, transcript, and report.
9. The browser presents a useful fallback if STT, LLM, or TTS is unavailable.

## 12. Immediate Work Order

Start in this order:

1. Fix session ID propagation in the frontend and prove it with an integration test.
2. Consolidate the streamed and non-streamed interview paths.
3. Add SQLite persistence and an interview report endpoint.
4. Define the four competency maps and migrate the question bank to tagged JSON records.
5. Build the question-quality test set before expanding the UI or adding more models.

This sequence protects the most valuable part of the product: a good question is not just fluent language. It is the right question for this candidate, asked at the right time, for a clear assessment reason, with the answer remembered and evaluated fairly.
