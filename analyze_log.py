import re
from collections import defaultdict

def analyze_log(log_file):
    """
    Analyzes a log file to calculate the count, average, and maximum probability for each dataset.

    Args:
        log_file (str): The path to the log file.
    """
    dataset_stats = defaultdict(lambda: {'probabilities': [], 'max_prob': 0.0, 'count': 0})

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'/mnt/f/([^/]+)/.* -> .*?_(\d+\.\d+)_.*\.wav', line)
            if match:
                dataset = match.group(1)
                probability = float(match.group(2))
                
                dataset_stats[dataset]['probabilities'].append(probability)
                if probability > dataset_stats[dataset]['max_prob']:
                    dataset_stats[dataset]['max_prob'] = probability
                dataset_stats[dataset]['count'] += 1

    print(f"{'Dataset':<20} {'Count':<10} {'Average Probability':<25} {'Maximum Probability'}")
    print("-" * 80)

    for dataset, stats in sorted(dataset_stats.items()):
        if stats['probabilities']:
            avg_prob = sum(stats['probabilities']) / len(stats['probabilities'])
            max_prob = stats['max_prob']
            count = stats['count']
            print(f"{dataset:<20} {count:<10} {avg_prob:<25.4f} {max_prob:<.4f}")

if __name__ == "__main__":
    analyze_log('prediction_2.log')