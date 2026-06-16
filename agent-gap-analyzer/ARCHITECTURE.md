# What it does
The main job of the gap-analyzer is to compare the user's input against that the AI engine actually produced. Specifically, it identifies any missing details or context that was not provided earlier in the conversation that caused the AI engine to produce a weak response or a response grounded in incomplete information. However, it does not assess the quality of the response; rather it just compares the AI's output with the user's request. This agent names the difference which is the gap that could have caused a misunderstanding such as not outputting what the user requested, or something that was only partially addressed but did not fully answer and assist the user. It could also be that the AI's response answered a completely different question than what the user asked for. 

## Why it exists

Engines hallucinate and drift most when there is not much context. The gap analyzer finds the specific missing pieces — background on the user, prior decisions, domain knowledge — so the coordinator can flag them rather than letting the engine fabricate or makeup its answers to fill the gaps.

A constraint checker needs to know what was asked or the context before it can judge whether the answer is appropriate and does not violate any rules. A gap analysis gives the downstream agents the context and details that they need to make accurate judgements regarding either fixing it or leaving it as it is.

## Input

The full context of conversation and the engine's response, which is passed by the coordinator
Receives the human's input verbatim from Agent 01 and the engine's proposed output

## Output

```json
{
  "agent": "gap-analyzer",
  "status": "complete" | "gaps-found",
  "gaps": ["description of missing context"]
}

## Position in the system

Runs in parallel. Its output is most useful when combined with the question generator — together they map what was not asked and what was not known.