"""Generate tests/fixtures/prefeval_tiny: a synthetic, PrefEval-format mini dataset.

Same file layout and keys as the real release (benchmark_dataset/explicit_preference/<topic>.json,
implicit_preference/{choice-based,persona-driven}/<topic>.json, filtered_inter_turns.json), but all
text is written here (no PrefEval content), so it can be committed. 10 topics x 4 items.
Run:  python tests/fixtures/make_prefeval_tiny.py
"""

import json
from pathlib import Path

TOPICS = {
    "travel_transportation": [
        ("I am afraid of flying, so I avoid planes entirely.", "Plan a five-day trip from London to Rome for me."),
        ("I get motion sick on boats and ferries.", "How should I get from Athens to the Greek islands?"),
        ("I refuse to rent cars that are not electric.", "Which rental car should I book for a road trip?"),
        ("I prefer overnight trains to save on hotels.", "What's the best way to travel across Europe cheaply?"),
    ],
    "lifestyle_dietary": [
        ("I am strictly vegan and eat no animal products.", "Suggest a dinner menu for my birthday party."),
        ("I have a severe peanut allergy.", "What snacks should I pack for a hiking trip?"),
        ("I avoid gluten because of coeliac disease.", "Recommend a bakery breakfast in Paris."),
        ("I don't drink alcohol at all.", "Plan an evening out with my colleagues."),
    ],
    "entertain_music_book": [
        ("I dislike heavy metal music.", "Make me a playlist for a long drive."),
        ("I only read physical books, never e-readers.", "What should I take to read on holiday?"),
        ("I love jazz from the 1950s.", "Recommend a concert to attend this weekend."),
        ("I can't stand horror novels.", "Suggest a book club pick for October."),
    ],
    "shop_technology": [
        ("I only use Apple devices and won't switch to Android.", "Which smartwatch should I buy?"),
        ("I refuse to buy anything with a subscription fee.", "Recommend software for editing photos."),
        ("I need gadgets that are repairable.", "What laptop should I get for university?"),
        ("I avoid smart speakers for privacy reasons.", "How can I set up a smart home?"),
    ],
    "lifestyle_fit": [
        ("I have a bad knee and cannot run.", "Give me a weekly cardio plan."),
        ("I hate going to crowded gyms.", "How can I build strength this year?"),
        ("I only have 20 minutes a day to exercise.", "Design a fitness routine for me."),
        ("I prefer exercising outdoors in nature.", "What workouts should I try this summer?"),
    ],
    "travel_hotel": [
        ("I only stay in hotels with a pool.", "Find me accommodation in Barcelona."),
        ("I avoid big chain hotels and prefer boutique ones.", "Where should I stay in Lisbon?"),
        ("I need wheelchair-accessible rooms.", "Book a hotel for my conference in Berlin."),
        ("I refuse to stay anywhere without free breakfast.", "Recommend a hotel near the beach."),
    ],
    "pet_ownership": [
        ("My dog is scared of loud noises.", "Plan a summer festival weekend for my family."),
        ("I am allergic to cats.", "Which pet should I adopt for my flat?"),
        ("I only buy grain-free pet food.", "What treats should I get for my puppy?"),
        ("I don't want pets that need daily walks.", "Suggest a low-maintenance companion animal."),
    ],
    "education_learning_styles": [
        ("I learn best through hands-on projects.", "How should I learn Python?"),
        ("I find video lectures boring.", "What's a good way to study history?"),
        ("I need material in short daily lessons.", "Help me learn Spanish this year."),
        ("I prefer visual diagrams over text.", "Explain how to understand statistics."),
    ],
    "shop_fashion": [
        ("I only wear sustainable, second-hand clothing.", "What should I wear to a wedding?"),
        ("I avoid leather products.", "Recommend shoes for the office."),
        ("I dislike bright colours in clothing.", "Help me pick a summer wardrobe."),
        ("I need clothes without wool, it makes me itch.", "What jumper should I buy for winter?"),
    ],
    "entertain_shows": [
        ("I don't watch anything with graphic violence.", "Recommend a series to binge this weekend."),
        ("I prefer documentaries over fiction.", "What should I watch on a rainy evening?"),
        ("I only watch shows with subtitles available.", "Suggest a foreign film."),
        ("I avoid reality TV completely.", "What TV shows are popular right now?"),
    ],
}

FILLER = [
    [("What is the capital of Australia?", "The capital of Australia is Canberra."),
     ("How many states does it have?", "Australia has six states and two mainland territories."),
     ("Which is the largest?", "Western Australia is the largest state by area.")],
    [("Write a haiku about autumn.", "Crisp leaves drift and fall / amber light on quiet paths / the year exhales slow."),
     ("Now one about winter.", "Snow hushes the street / breath clouds in the silver dawn / the world holds its pause."),
     ("Thanks, that's lovely.", "You're welcome. Happy to write more any time.")],
    [("Convert 5 miles to kilometres.", "5 miles is about 8.05 kilometres."),
     ("And 10 pounds to kilograms?", "10 pounds is about 4.54 kilograms."),
     ("What about 70 Fahrenheit in Celsius?", "70 °F is about 21.1 °C.")],
]


def main() -> None:
    root = Path(__file__).parent / "prefeval_tiny" / "benchmark_dataset"
    for sub in ["explicit_preference", "implicit_preference/choice-based", "implicit_preference/persona-driven"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    for topic, rows in TOPICS.items():
        explicit = [{"preference": p, "question": q, "explanation": "synthetic"} for p, q in rows]
        choice = [dict(e, implicit_query=f"Options for: {e['question']}", options=["A", "B", "C", "D"],
                       conversation={"query": e["question"], "assistant_options": "1. A 2. B 3. C 4. D",
                                     "user_selection": f"I'll pick A; {e['preference']}",
                                     "assistant_acknowledgment": "Noted."}, aligned_op="A") for e in explicit]
        persona = [dict(e, persona="A synthetic persona",
                        conversation={"0": {"user": f"By the way, {e['preference']}", "assistant": "Understood."}})
                   for e in explicit]
        (root / "explicit_preference" / f"{topic}.json").write_text(json.dumps(explicit, indent=1))
        (root / "implicit_preference/choice-based" / f"{topic}.json").write_text(json.dumps(choice, indent=1))
        (root / "implicit_preference/persona-driven" / f"{topic}.json").write_text(json.dumps(persona, indent=1))
    convs = []
    for i, ex in enumerate(FILLER):
        turns = []
        for u, a in ex:
            turns += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        convs.append({"conversation_id": f"tiny{i}", "model": "synthetic", "conversation": turns,
                      "turn": len(ex), "language": "English", "redacted": False})
    (root / "filtered_inter_turns.json").write_text(json.dumps(convs, indent=1))


if __name__ == "__main__":
    main()
