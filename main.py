"""DiagKnows command-line menu.

Run:  python main.py            (plain-language results)
      python main.py --details  (also show distances and neighbor counts)
Builds the database on first launch (requires data/Symptom2Disease.csv).
"""

import os
import sys
import textwrap

import diagknows as dk
import explain

WIDTH = 78
SHOW_DETAILS = "--details" in sys.argv[1:]
USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")

MATCH_LABELS = {
    "high": ("●●●", "Strong match"),
    "medium": ("●●○", "Possible match"),
    "low": ("●○○", "Weak match"),
}

MENU = "\n[1] Check symptoms   [2] Add a case   [3] Stats   [4] Quit"


def style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(text: str) -> str:
    return style(text, "1")


def dim(text: str) -> str:
    return style(text, "2")


def display_name(condition: str) -> str:
    # The dataset mixes "Common Cold" with "allergy"; capitalize for display only.
    return condition[:1].upper() + condition[1:]


def shorten(text: str, width: int) -> str:
    short = textwrap.shorten(text, width=width, placeholder="")
    if short == " ".join(text.split()):
        return short
    return short.rstrip(" .,;:") + "…"


def wrap(text: str, indent: str) -> str:
    return textwrap.fill(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent)


def ask(prompt: str) -> str:
    return input(prompt).strip()


def print_disclaimer(text: str) -> None:
    print("\n" + dim(wrap(text, indent="")))


def describe_symptoms() -> None:
    query = ask("\nDescribe how you feel: ")
    if not query:
        print("Nothing entered.")
        return

    result = dk.rank_conditions(query)
    print("\n" + "─" * WIDTH)

    if result["status"] == "emergency":
        print(bold(style("⚠  SEEK EMERGENCY CARE", "31")))
        print(wrap(result["message"], indent=""))
        if SHOW_DETAILS:
            e = result["emergency"]
            print(dim(f"   matched red flag (distance {e['distance']:.2f}): {e['matched_red_flag']}"))

    elif result["status"] == "no_match":
        print(bold("No confident match."))
        print("Try describing your symptoms in more detail: where it hurts, what it feels")
        print("like, and how long it's been going on.")
        if SHOW_DETAILS:
            print(dim(f"   closest distance {result['best_distance']:.2f} (cutoff {dk.MATCH_CUTOFF})"))

    else:
        print(bold("Your symptoms most resemble:"))
        for i, c in enumerate(result["conditions"], start=1):
            meter, label = MATCH_LABELS[c["confidence"]]
            # Pad before styling so color codes don't throw off the alignment.
            print(f"\n {i}. {bold(f'{display_name(c['condition']):<32}')} {meter}  {label}")
            print(f"    Doctor:  {c['doctor']}")
            ex = c["examples"][0]
            source = " (added by a user)" if ex["source"] == "user" else ""
            print(f"    Similar: \"{shorten(ex['text'], WIDTH - 15 - len(source))}\"{source}")
            if SHOW_DETAILS:
                print(dim(f"    {c['votes']}/{dk.N_NEIGHBORS} neighbors, {c['share']:.0%} of score, "
                          f"closest distance {c['best_distance']:.2f}"))

        if all(c["confidence"] == "low" for c in result["conditions"]):
            print("\nThese are all weak matches. More detail usually helps: where it hurts,")
            print("what it feels like, and how long it's been going on.")

        explanation = explain.explain(result)
        if explanation:
            print("\n" + bold("Why these came up") + dim(" (AI-written from the similar cases)"))
            print(wrap(explanation, indent="    "))

    print_disclaimer(result["disclaimer"])
    print("─" * WIDTH)


def add_new_case() -> None:
    description = ask("\nDescribe the symptoms for the new case: ")
    condition = ask("Condition (e.g. Migraine, or a new one): ")
    try:
        case_id = dk.add_case(description, condition)
    except ValueError as e:
        print(e)
        return
    print(f"Added. It's searchable now. {dim('(id: ' + case_id + ')')}")
    print_disclaimer(dk.DISCLAIMER)


def show_stats() -> None:
    s = dk.stats()
    print(f"\n{bold('Database')}: {s['total']} cases ({s['user_added']} added by users)\n")
    for condition, count in s["per_condition"].items():
        print(f"  {count:>4}  {display_name(condition)}")


def main() -> None:
    dk.ensure_database()
    print("\n" + bold("DiagKnows") + " - describe how you feel and see which conditions it resembles.")
    if not explain.is_enabled():
        print(dim("(AI explanations off: no OPENAI_API_KEY set. Everything else works.)"))

    actions = {"1": describe_symptoms, "2": add_new_case, "3": show_stats}
    while True:
        print(MENU)
        try:
            choice = ask("> ")
            if choice == "4":
                print("Goodbye.")
                break
            if choice in actions:
                actions[choice]()
            else:
                print("Please enter 1, 2, 3, or 4.")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
