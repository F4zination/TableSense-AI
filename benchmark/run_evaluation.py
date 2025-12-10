import json
import pandas as pd


def style_correctness(val):
    """
    Color coding: Green for Correct, Red for Incorrect.
    """
    if val is True:
        return 'background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb'
    elif val is False:
        return 'background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb'
    return 'color: gray'


def compare_runs_visually(file_path1, file_path2, output_file='comparison_report.html'):
    # 1. Load Data
    try:
        with open(file_path1, 'r') as f1:
            data1 = json.load(f1)
        with open(file_path2, 'r') as f2:
            data2 = json.load(f2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    map1 = {item['index']: item for item in data1}
    map2 = {item['index']: item for item in data2}
    all_indices = sorted(set(map1.keys()) | set(map2.keys()))

    rows = []

    # --- Statistics Counters ---
    stats = {
        "both_correct": 0,
        "both_incorrect": 0,
        "r1_incorrect_r2_correct": 0,
        "r1_correct_r2_incorrect": 0
    }

    # 2. Iterate and Compare
    for idx in all_indices:
        item1 = map1.get(idx)
        item2 = map2.get(idx)

        # Determine Correctness
        # We convert to string to ensure '99' matches 99
        r1_is_correct = False
        if item1:
            r1_is_correct = str(item1.get('pred')) == str(item1.get('ground_truth'))

        r2_is_correct = False
        if item2:
            r2_is_correct = str(item2.get('pred')) == str(item2.get('ground_truth'))

        # Update Statistics
        if r1_is_correct and r2_is_correct:
            stats["both_correct"] += 1
        elif not r1_is_correct and not r2_is_correct:
            stats["both_incorrect"] += 1
        elif not r1_is_correct and r2_is_correct:
            stats["r1_incorrect_r2_correct"] += 1
        elif r1_is_correct and not r2_is_correct:
            stats["r1_correct_r2_incorrect"] += 1

        # Add row to list
        question_text = item1['question'] if item1 else (item2['question'] if item2 else "Unknown")
        rows.append({
            "Index": idx,
            "Question": question_text,
            "Run 1 Result": r1_is_correct,
            "Run 2 Result": r2_is_correct
        })

    # 3. Print Statistics to Console
    print("-" * 40)
    print("COMPARISON STATISTICS")
    print("-" * 40)
    print(f"Both Incorrect:                 {stats['both_incorrect']}")
    print(f"Run 1 Incorrect / Run 2 Correct: {stats['r1_incorrect_r2_correct']}")
    print(f"Run 1 Correct / Run 2 Incorrect: {stats['r1_correct_r2_incorrect']}")
    print(f"Both Correct:                   {stats['both_correct']}")
    print("-" * 40)

    # 4. Create HTML Report
    df = pd.DataFrame(rows)

    # Apply styling
    styled_df = df.style.map(style_correctness, subset=["Run 1 Result", "Run 2 Result"]) \
        .format({
        "Run 1 Result": lambda x: "Correct" if x else "Incorrect",
        "Run 2 Result": lambda x: "Correct" if x else "Incorrect"
    }) \
        .set_table_styles([
        {'selector': 'th',
         'props': [('font-size', '12pt'), ('text-align', 'left'), ('background-color', '#f0f0f0'), ('padding', '8px')]},
        {'selector': 'td', 'props': [('padding', '8px'), ('border-bottom', '1px solid #ddd')]},
        {'selector': 'table',
         'props': [('border-collapse', 'collapse'), ('width', '100%'), ('font-family', 'Arial, sans-serif')]}
    ]) \
        .hide(axis="index")

    # Add Summary Header to HTML
    summary_html = f"""
    <html>
    <head><title>Comparison Report</title></head>
    <body style="font-family: Arial, sans-serif; padding: 20px;">
        <h2>Comparison Summary</h2>
        <table style="margin-bottom: 20px; border: 1px solid #ddd; padding: 10px;">
            <tr><td style="padding:5px;"><strong>Both Incorrect:</strong></td><td style="padding:5px;">{stats['both_incorrect']}</td></tr>
            <tr><td style="padding:5px;"><strong>Run 1 Incorrect / Run 2 Correct:</strong></td><td style="padding:5px; background-color: #d4edda;">{stats['r1_incorrect_r2_correct']}</td></tr>
            <tr><td style="padding:5px;"><strong>Run 1 Correct / Run 2 Incorrect:</strong></td><td style="padding:5px;">{stats['r1_correct_r2_incorrect']}</td></tr>
            <tr><td style="padding:5px;"><strong>Both Correct:</strong></td><td style="padding:5px;">{stats['both_correct']}</td></tr>
        </table>
        <hr>
    """

    # Save final file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary_html + styled_df.to_html() + "</body></html>")

    print(f"Report saved to: {output_file}")


# --- Run ---
#compare_runs_visually('run_serialization_tabmwp/examples_TabMWPSelection.json', 'run_smolagent_tabmwp/examples_TabMWPSelection.json')

compare_runs_visually('run_serialization_wikiquestions/examples_WikiTableQuestionsSelection.json', 'run_smolagent_wikitablequestion/examples_WikiTableQuestionsSelection.json')
