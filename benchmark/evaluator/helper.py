def average_results(complete_results):
    from collections import defaultdict

    # First, organize all results per dataset_name
    dataset_metrics = defaultdict(list)

    for run in complete_results:
        for dataset_result in run:
            name = dataset_result["dataset_name"]
            dataset_metrics[name].append(dataset_result)

    averaged_results = []
    for name, results in dataset_metrics.items():
        n = len(results)
        avg_result = {'dataset_name': name}

        # Average Exact match
        exact_matches = [float(r['Exact match']['exact_match']) for r in results]
        avg_result['Exact match'] = {'exact_match': sum(exact_matches) / n}

        # Average BERTScore
        bert_metrics = ['precision', 'recall', 'f1']
        avg_result['BERTScore'] = {
            metric: sum(float(r['BERTScore'][metric]) for r in results) / n
            for metric in bert_metrics
        }

        # Average Rogue Score
        rouge_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        avg_result['Rogue Score'] = {
            metric: sum(float(r['Rogue Score'][metric]) for r in results) / n
            for metric in rouge_metrics
        }

        averaged_results.append(avg_result)

    return averaged_results
