import argparse
import numpy as np
import os
import pickle

# Calculates forgetting statistics per example
def compute_forgetting_statistics(diag_stats, npresentations):
    presentations_needed_to_learn = {}
    unlearned_per_presentation = {}
    margins_per_presentation = {}
    first_learned = {}

    for example_id, example_stats in diag_stats.items():
        # Skip 'train' and 'test' keys of diag_stats
        if not isinstance(example_id, str):
            # Forgetting event is a transition in accuracy from 1 to 0
            presentation_acc = np.array(example_stats[1][:npresentations])
            transitions = presentation_acc[1:] - presentation_acc[:-1]

            # Find all presentations when forgetting occurs
            if len(np.where(transitions == -1)[0]) > 0:
                unlearned_per_presentation[example_id] = np.where(
                    transitions == -1)[0] + 2
            else:
                unlearned_per_presentation[example_id] = []

            # Find number of presentations needed to learn example, 
            # e.g. last presentation when acc is 0
            if len(np.where(presentation_acc == 0)[0]) > 0:
                presentations_needed_to_learn[example_id] = np.where(
                    presentation_acc == 0)[0][-1] + 1
            else:
                presentations_needed_to_learn[example_id] = 0

            # Find the misclassication margin for each presentation of the example
            margins_per_presentation[example_id] = np.array(
                example_stats[2][:npresentations])

            # Find the presentation at which the example was first learned, 
            # e.g. first presentation when acc is 1
            if len(np.where(presentation_acc == 1)[0]) > 0:
                first_learned[example_id] = np.where(
                    presentation_acc == 1)[0][0]
            else:
                first_learned[example_id] = np.nan

    #print(f"Number of examples processed: {len(presentations_needed_to_learn), len(unlearned_per_presentation)}")
    return presentations_needed_to_learn, unlearned_per_presentation, margins_per_presentation, first_learned


# Sorts examples by number of forgetting counts during training, in ascending order
def sort_examples_by_forgetting(unlearned_per_presentation_all,
                                first_learned_all, npresentations):
    # Initialize lists
    example_original_order = []
    example_stats = []

    for example_id in unlearned_per_presentation_all[0].keys():
        # Add current example to lists
        example_original_order.append(example_id)
        example_stats.append(0)

        # Iterate over all training runs to calculate the total forgetting count for current example
        for i in range(len(unlearned_per_presentation_all)):
            # Get all presentations when current example was forgotten during current training run
            stats = unlearned_per_presentation_all[i][example_id]

            # If example was never learned during current training run, add max forgetting counts
            if np.isnan(first_learned_all[i][example_id]):
                example_stats[-1] += npresentations
            else:
                example_stats[-1] += len(stats)

    print('Number of unforgettable examples: {}'.format(
        len(np.where(np.array(example_stats) == 0)[0])))
    
    sorted_indices = np.argsort(example_stats)
    ordered_examples = np.array(example_original_order)[sorted_indices]
    ordered_values = np.sort(example_stats)
    

    ordered_list = [(int(idx), int(count)) for idx, count in zip(ordered_examples, ordered_values)]

    print(len(ordered_list))
    print(ordered_list)

    return np.array(example_original_order)[np.argsort(example_stats)] \
            , np.sort(example_stats)



# Checks whether a given file name matches a specified filename
def check_filename(fname, target_fname):
    return fname == target_fname

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--input_fname', type=str, required=True, help='input filename, e.g., abcd__stats_dict.pkl')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)

    args = parser.parse_args()
    print(args)

    # Initialize lists to collect forgetting statistics per example across multiple training runs
    unlearned_per_presentation_all, first_learned_all = [], []

    for d, _, fs in os.walk(args.input_dir):
        for f in fs:
            # Find the files that match input_fname and compute forgetting statistics
            if f == args.input_fname and f.endswith('stats_dict.pkl'):
                print('including file: ' + f)

                # Load the dictionary compiled during training run
                with open(os.path.join(d, f), 'rb') as fin:
                    loaded = pickle.load(fin)

                # Compute the forgetting statistics per example for training run
                _, unlearned_per_presentation, _, first_learned = compute_forgetting_statistics(
                    loaded, args.epochs)

                unlearned_per_presentation_all.append(unlearned_per_presentation)
                first_learned_all.append(first_learned)

    #print(unlearned_per_presentation_all, first_learned_all)

    if len(unlearned_per_presentation_all) == 0:
        print('No input files found in {} that match {}'.format(args.input_dir, args.input_fname))
    else:
        # Sort examples by forgetting counts in ascending order, over one or more training runs
        ordered_examples, ordered_values = sort_examples_by_forgetting(
            unlearned_per_presentation_all, first_learned_all, args.epochs)

        # Save sorted output
        output_path = os.path.join(args.output_dir, args.output_name)
        if not output_path.endswith('.pkl'):
            output_path += '.pkl'
        
        print(f"## Number of ordered examples: {len(ordered_examples)}")



        with open(output_path, 'wb') as fout:
            pickle.dump({
                'indices': ordered_examples,
                'forgetting counts': ordered_values
            }, fout)
