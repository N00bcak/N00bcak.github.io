---
title:  "CTFs are Dead. Long Live CTFs!"
date:   2026-06-14
---

Honestly, I'm sick and tired of hearing people say "CTFs are dead" ([do](https://blog.kek.cx/posts/slop-is-all-you-need/), [re](https://samuzora.com/posts/20-years), [mi](https://kabir.au/blog/the-ctf-scene-is-dead), [fa](https://lobste.rs/s/duyp69/ctf_scene_is_dead), [so](https://www.linkedin.com/posts/zhang-zeyu_are-ctfs-dead-its-been-a-year-since-i-last-activity-7466762589629206528-1olw), ...).

What makes these blogposts so **frustrating** is that everything happening in CTFs now seems to be telling you that... they're actually right.

CTFs used to be a Thinking Man's Sport (tm), and now it has degenerated into a game of "who-can-spam-their-clanker-the-fastest".

Anyone who still insists that nothing has changed is delusional. The old open scoreboard is compromised. A team with money, taste, and a half-decent agent harness can now erase entire categories of challenges that used to take human beings hours of work. Beginner and intermediate challenges are especially exposed. A lot of the old signals are polluted. A first blood no longer necessarily means someone had the insight first. A high placement no longer necessarily means the team understood the most. Sometimes it means they had the better model, the better harness, the better prompt cache, the better luck, or simply the weaker sense of shame.

That is real.

But "CTFs are dead" is not a diagnosis. It is a resignation letter.

It mistakes a broken format for a dead culture. It sees a scoreboard rotting and concludes that the entire discipline must be buried with it. It watches AI eat the shallow end of the pool and decides the ocean has dried up.

Worse, it quietly absolves us.

If CTFs are dead because AI killed them, then nobody has to do anything. Challenge authors can stop trying. Organisers can keep shipping the same format with a new coat of CSS. Players can either clank everything and feel hollow, or refuse to clank and feel cheated. The rest of us can sit around writing elegies for human ingenuity while doing almost nothing to defend the conditions under which human ingenuity actually appears.

That is the part I reject.

AI did not kill CTFs. We are letting CTFs die because we are too lazy, too scared, and too incurious to adapt the game around the new reality.

## The old scoreboard is not sacred

The first mistake is treating the scoreboard as if it were the soul of CTF.

It never was.

The scoreboard was a compression algorithm. It took a large, messy, beautiful process — reading code, misunderstanding systems, forming hypotheses, writing broken solve scripts, chasing false leads, discovering weird machine behaviour, and finally seeing the flag — and compressed it into a number.

That number was useful because, for a while, it correlated decently with the thing we cared about. If you solved more hard challenges faster than everyone else, you were probably very good. Not always. There was always flag sharing, guessing, infra jank, team-size imbalance, unintendeds, sleep deprivation, and metagame nonsense. But the lie was small enough to be productive.

AI made the lie bigger.

Now the scoreboard often measures a composite of human skill, model capability, orchestration, API spend, automation hygiene, and willingness to let a machine perform the actual learning loop. It has not become meaningless, but it has become ambiguous. That ambiguity is poison if we keep pretending the old interpretation still applies.

So stop pretending.

The answer is not to worship the old scoreboard harder. The answer is to admit that one scoreboard cannot represent every mode of play anymore.

We need separate tracks.

Not as a moral purity ritual. As basic measurement hygiene.

A CTF that allows unrestricted AI and a CTF that rewards unaided human solving are no longer the same game. They may share challenges, infrastructure, flags, and Discord servers, but they measure different skills. One measures human-machine systems: decomposition, tool design, agent steering, verification, and automation. The other measures human technical reasoning under constraints. Both can be interesting. Both can be legitimate. They should not silently share the same leaderboard and then act surprised when everyone feels cheated.

We already understand this everywhere else.

We separate individual and team rankings. We separate junior and open categories. We separate on-site finals from online qualifiers. We separate university, high school, professional, and sometimes even country-specific brackets. We do this because context matters. Constraints matter. A ranking is only honest when the players are playing comparable games.

AI use is now one of those constraints.

A serious CTF in 2026 should be able to say, clearly:

* this is the open track;
* this is the human-only or human-primary track;
* this is the agentic track;
* this is the beginner learning track;
* this is the prize pool attached to each;
* this is what evidence, honour, auditing, or proctoring we expect;
* this is what we are actually trying to measure.

This does not solve everything. Rules can be violated. Remote competitions cannot perfectly enforce honesty. Some people will cheat.

That is not an argument against having rules. It is an argument against childish expectations.

We have never had perfect enforcement. We still had norms. We still had categories. We still had shame. We still had trust. We still had enough structure for the game to function most of the time.

If we can enforce nothing perfectly, then we should enforce what we can, design incentives around what we value, and stop using impossibility as an excuse for indolence.

## Stop making anti-AI challenges that are also anti-human

The second mistake is just as bad: trying to stop clanking by making challenges miserable.

You have seen these challenges. Maybe you have written one. Maybe I have.

The challenge whose entire purpose is to trick the model with prompt-injection bait. The challenge where the intended path is hidden behind arbitrary guessing. The challenge that depends on a custom file format with no conceptual payoff. The challenge that is "AI-resistant" because it is incoherent. The challenge that punishes every normal debugging instinct. The challenge that is not hard because the system is deep, but hard because the author made it hostile to understanding.

This is not saving CTFs.

This is salting the earth and calling it agriculture.

A challenge that is unplayable for humans is not a victory over AI. It is a surrender wearing a fake moustache. It says: "We could not design something worth thinking about, so we designed something painful to automate."

That route ends badly. It produces uglier challenges, angrier players, worse writeups, and no durable defence. The models get better. The prompts get better. The harnesses get better. The only thing left behind is a generation of CTFs nobody enjoyed.

Anti-AI design cannot mean anti-reasoning design.

The goal is not to make the model hallucinate. The goal is to make the human matter.

Those are different.

A good modern challenge should not ask, "How do I stop ChatGPT from solving this?" It should ask:

> Where, in this challenge, does a human being have to make a real judgment?

That judgment can take many forms.

It can be domain knowledge: knowing enough about compilers, kernels, cryptographic protocols, browser internals, RF, hardware, databases, distributed systems, or weird historical software to even recognise the shape of the problem.

It can be taste: deciding which of five plausible attack surfaces is actually worth pursuing.

It can be experimental discipline: building a harness, isolating variables, noticing that a crash is not the bug but a symptom.

It can be creativity: connecting two behaviours that look unrelated until they suddenly explain each other.

It can be strategic steering: using AI as a fast assistant while refusing to let it define the entire search space.

It can be adversarial scepticism: reading a generated solve and saying, "No. This is plausible nonsense. The invariant is wrong."

This is where human thinking still lives. Not in typing every line of boilerplate by hand. Not in pretending objdump is sacred. Not in refusing help from tools because suffering is virtuous. Human thinking lives in choosing what matters, recognising what is false, and changing direction when the obvious path is a trap.

If your challenge has no place for that, AI did not ruin it. It exposed it.

## The problem is not that AI is too strong. The problem is that many challenges are too thin.

This is the uncomfortable part.

A lot of CTF challenges were always thin.

They were still useful. They taught syntax, patterns, techniques, and basic exploitation muscle memory. They gave beginners a ladder. They gave intermediate players a way to build speed. They gave organisers a way to fill a board.

But many of them depended on friction that was never the essence of security thinking.

Finding the right tool. Remembering the right command. Translating a known vulnerability class into a solve script. Googling the right error. Writing the same pwntools scaffold for the 500th time. Recognising the same RSA footgun. Seeing the same pickle sink. Running the same angr ritual. Copying the same JWT trick. Chaining the same web primitives.

This work was not worthless. It was how many of us learned.

But we should be honest: a lot of it was pattern matching surrounded by toil.

LLMs are very good at pattern matching surrounded by toil.

That does not mean CTFs are over. It means the floor moved. The part of CTFing that was mostly recall, syntax, glue code, and standard technique execution is now heavily automatable. Good. Let it be automatable. That was never the highest form of the craft.

The question is what we build above it.

If we keep writing challenges whose core idea can be recovered from three familiar keywords and solved by stitching together known snippets, then yes, those challenges will be clanked. They should be. They are now table stakes.

But there is still a large space of challenges that require more than that:

* challenges based on fresh or obscure domain mechanics, not merely obscure trivia;
* challenges with systems large enough that the key is modelling behaviour, not spotting a single sink;
* challenges where the exploit path depends on understanding business logic, state transitions, or emergent interactions;
* challenges that require building small tools, measurements, or visualisations before the bug becomes clear;
* challenges where generated code is useful but generated reasoning is insufficient;
* challenges that reward players for falsifying hypotheses, not merely producing a payload;
* challenges where the "solve" is not a known exploit template but a sequence of decisions.

This does not mean every challenge must become a three-day research project. Beginner CTFs still matter. Small challenges still matter. Not every problem needs a thesis attached to it.

But if a challenge is intended to distinguish skilled human players in an AI-saturated environment, then it needs depth somewhere. Not arbitrary depth. Real depth. Domain depth. System depth. Conceptual depth. Depth that teaches something when the player digs into it.

The bar has moved. We can complain about that, or we can start authoring for it.

## "Just ban AI" is not stupid. It is just incomplete.

There is a lazy response to AI bans that goes like this:

> You cannot enforce it, so do not try.

This sounds practical. It is often just defeatism with better posture.

A human-only track has value even if enforcement is imperfect. Especially for local CTFs, student CTFs, training events, on-site finals, invite-only contests, and communities with actual social trust, restrictions can work well enough to matter. Not perfectly. Enough.

The goal is not to create a mathematically pure environment. The goal is to preserve a mode of play.

Chess did not become pointless because engines exist. Speedrunning did not become pointless because tool-assisted speedruns exist. Programming contests did not become pointless because Stack Overflow existed, and they did not immediately become pointless because Copilot existed either. These communities survived by drawing lines, building categories, and attaching prestige to the modes of play they cared about.

CTFs can do the same.

A no-AI or limited-AI track should not be framed as nostalgia. It should be framed as a discipline.

You are choosing to practise human exploitation skill. You are choosing to struggle. You are choosing to develop the instincts that make you useful even when the model is wrong. You are choosing the harder path because the harder path changes you.

That deserves recognition.

Give it a separate leaderboard. Give it separate prizes. Give it a badge. Require writeups. Require solve logs for prize eligibility. Use random interviews for top teams. Run on-site finals where possible. Ask finalists to explain their exploit chain live. Make the honour system explicit and attach reputation to it.

Will someone still cheat?

Yes.

Someone always cheats.

Design anyway.

The alternative is to declare that because integrity is imperfect, integrity is obsolete. That is cowardice disguised as realism.

## The agentic track should exist too

Banning AI everywhere is not the answer either.

There is a real and interesting skill in building systems that solve CTF challenges with AI. It is not the same skill as traditional CTF solving, but it is not nothing. Decomposition, orchestration, tool integration, context management, verifier design, exploit validation, sandboxing, cost control, and failure recovery are all legitimate engineering problems.

Pretending otherwise is silly.

An agentic CTF track could be excellent if it stopped freeloading on the prestige of human CTF and became honest about what it is.

Make the objective explicit. Score not only flags, but cost, reproducibility, autonomy, and correctness. Require teams to submit harnesses. Reward systems that produce readable solve reports. Penalise unverifiable garbage. Include challenges designed to expose model weaknesses safely: false positives, brittle assumptions, poisoned documentation, misleading decompiler output, flaky services, long-horizon state, and tasks where the model must ask for human steering.

That would be a new game.

It might even become a better benchmark for practical AI-assisted security than the current mess, where everyone quietly runs private agent swarms against a board that was never designed to measure them.

The important thing is honesty.

A bicycle is not cheating in a bicycle race. It is cheating in a footrace. A CTF team using an agent swarm is not inherently shameful. A CTF team using an agent swarm while pretending to be engaged in the same human contest as everyone else is the problem.

Different games. Different rankings. Different prestige.

This is not complicated. We are making it complicated because we do not want to give up the convenience of one universal scoreboard.

## Average players are not powerless

The most damaging myth right now is that only organisers, top teams, or model labs can do anything.

False.

Average CTF players can change the incentive structure faster than they think.

They can ask organisers for separate tracks instead of merely complaining after the event. They can publish writeups that distinguish human reasoning from AI assistance. They can refuse to treat a clanked solve as equivalent to a worked solve. They can praise challenges that reward understanding, not just flags. They can give feedback when anti-AI gimmicks make a challenge worse. They can run small local events with explicit norms. They can form teams where learning is valued over scoreboard vanity. They can share tooling that helps humans learn instead of tooling that merely converts API credits into flags.

Most importantly, they can stop lying to themselves.

If you clanked a challenge, say you clanked it.

There is no need for melodrama. There is no need for confession. Just be precise. "I solved this manually." "I used AI for boilerplate." "I used AI to explain a primitive." "An agent found the exploit path." "I verified the solve but did not discover it."

That vocabulary matters. It restores resolution to a scene that is currently becoming blurry.

The same applies to organisers. Stop asking "AI allowed?" as a throwaway rules-page line nobody believes. Ask what kind of event you are running. Ask what kind of player you want to reward. Ask what kind of writeups you want to read afterwards. Ask what kind of skills the leaderboard should signal.

If you want a human-first learning CTF, design for that. If you want an open cyber automation brawl, design for that. If you want both, separate them.

But do not run the same old board, watch it get clanked, and then publish a postmortem about the death of thought.

That is not tragedy. That is negligence.

## Human thinking is not dead

The phrase "death of human thinking" is rhetorically powerful and technically lazy.

Human thinking is not dead because a model can solve a medium web challenge. Human thinking is not dead because an agent can write a pwntools script. Human thinking is not dead because the old ladder is broken.

Human thinking dies when we stop demanding it from ourselves.

It dies when challenge authors ship shallow problems because "the models will solve everything anyway." It dies when organisers refuse to update formats because "enforcement is hard." It dies when players let the model take the wheel before they have even read the source. It dies when communities reward flags without caring how they were obtained. It dies when we confuse difficulty with opacity, and adaptation with surrender.

That is the death I am afraid of.

Not the machine becoming strong.

Us becoming passive.

Because the machine is not going away. The models will improve. The harnesses will improve. The cost curves will change. The easy objections will age badly. Any plan that depends on AI getting worse is not a plan.

But any plan that assumes humans have no role left is just as unserious.

The future of CTFs is not "no AI" or "only AI." It is a set of clearer games built around clearer values:

* human-only competition for developing and measuring unaided skill;
* AI-assisted competition for realistic modern workflow;
* agentic competition for automation and orchestration;
* learning-first events where the scoreboard is secondary or redesigned;
* challenge formats that reward explanation, verification, creativity, and depth;
* community norms that stop treating every flag as the same kind of achievement.

This is not science fiction. It is administration, taste, and will.

The work is boring in places. It means writing better rules. Building better infra. Reviewing writeups. Splitting prize pools. Saying no to sponsors who only want big numbers. Admitting that some old metrics are no longer clean. Training new authors. Giving players language for how they used AI. Creating events where the desired behaviour is easier than the degenerate behaviour.

None of this is beyond us.

The CTF community has built absurd things for fun. We have reverse engineered dead software, exploited imaginary kernels, solved bespoke crypto systems, escaped sandboxes, glitched hardware, and found bugs in systems whose authors swore there were none. We can manage a leaderboard with more than one column.

## CTFs are dead. Long live CTFs.

The old CTF was not perfect. We should stop pretending it was.

But it had something worth preserving: the ladder, the craft, the shared struggle, the moment where a person understands a system deeply enough to break it.

AI threatens that only if we flatten every mode of play into the same scoreboard and then act confused when the cheapest strategy wins.

So no, I am not ready to mourn CTFs.

I am ready to bury the lazy version of them.

Bury the single scoreboard that pretends human solvers and agent swarms are equivalent. Bury the anti-AI gimmick challenge that nobody enjoys. Bury the fatalism that calls itself realism. Bury the nostalgia that thinks refusing to change is the same as defending human skill.

Then build.

Build tracks that measure what they claim to measure. Build challenges with real depth. Build AI-assisted formats that are honest about being AI-assisted. Build human-first spaces where beginners can struggle without being humiliated by someone's token budget. Build agentic competitions that evaluate agents properly instead of letting them parasitise human ones. Build norms around disclosure. Build prestige around understanding, not just submission time.

CTFs are dying because we are letting them die.

Not accidentally. Not innocently. We are letting them die every time we choose the easy posture: denial, despair, or gimmickry.

The better posture is harder.

It says: the old game is broken, and we are still responsible for the next one.

That is not the death of CTFs.

That is the start of the next flag.